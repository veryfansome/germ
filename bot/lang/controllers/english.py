from datasets import Dataset
from datetime import datetime
from starlette.concurrency import run_in_threadpool
from traceback import format_exc
import aiohttp
import asyncio
import inflect
import os

from bot.graph.control_plane import (CodeBlockMergeEventHandler, ControlPlane, ParagraphMergeEventHandler,
                                     SentenceMergeEventHandler)
from bot.lang.dependencies import wordnet_lemmatizer
from bot.lang.parsers import get_html_soup, strip_html_elements
from bot.lang.patterns import naive_sentence_end_pattern
from observability.logging import logging
from settings import germ_settings

logger = logging.getLogger(__name__)

infect_eng = inflect.engine()

NOUN_LABELS = {"NN", "NNS", "NNP", "NNPS"}
VERB_LABELS = {"VB", "VBD", "VBG", "VBP", "VBN", "VBZ"}
STATE_OF_BEING_VERBS = {"am", "are", "be", "been", "being", "is", "was", "were"}
SUBORDINATING_CONJUNCTIONS = {"after", "although", "because", "before", "if", "once", "since", "that", "though",
                              "unless", "until", "when", "while"}


class EnglishController(CodeBlockMergeEventHandler, ParagraphMergeEventHandler, SentenceMergeEventHandler):
    def __init__(self, control_plane: ControlPlane, interval_seconds: int = 9):
        self.control_plane = control_plane
        self.interval_seconds = interval_seconds
        self.labeled_examples_conll = []
        self.labeled_examples_ud = []
        self.unlabeled_sentences = []

        self.adjectives_added = set()
        self.noun_classes_added = set()
        self.nouns_added = set()

    async def add_det_noun(self, det: str, noun: str, sentence_id: int):
        if det in {"a", "an"}:
            if noun not in self.noun_classes_added:
                await self.control_plane.add_noun_class(noun)
                self.noun_classes_added.add(noun)
        else:
            if noun not in self.nouns_added:
                await self.control_plane.add_noun(noun, sentence_id)
                self.nouns_added.add(noun)

    async def dump_labeled_exps(self):
        async_tasks = []
        conll_exps = []
        ud_exps = []
        if self.labeled_examples_conll:
            while self.labeled_examples_conll:
                conll_exps.append(self.labeled_examples_conll.pop(0))
            async_tasks.append(asyncio.create_task(run_in_threadpool(dump_labeled_exps, conll_exps, "conll")))
        if self.labeled_examples_ud:
            while self.labeled_examples_ud:
                ud_exps.append(self.labeled_examples_ud.pop(0))
            async_tasks.append(asyncio.create_task(run_in_threadpool(dump_labeled_exps, ud_exps, "ud")))
        await asyncio.gather(*async_tasks)

    async def label_sentence(self, sentence: str, sentence_id: int, sentence_context):
        conll_token_labels, ud_token_labels = await asyncio.gather(
            get_conll_token_classifications(sentence),
            get_ud_token_classifications(sentence),
        )
        logger.info(f"sentence_context: \t{sentence_context}")
        logger.info(f"conll labels: sentence_id={sentence_id}\n" + (
            "\n".join([f"{head}\t{labels}" for head, labels in conll_token_labels.items()])))
        logger.info(f"ud labels: sentence_id={sentence_id}\n" + (
            "\n".join([f"{head}\t{labels}" for head, labels in ud_token_labels.items()])))

        is_conll_ud_aligned = True
        conll_token_len = len(conll_token_labels["tokens"])
        ud_token_len = len(ud_token_labels["tokens"])
        if conll_token_len != ud_token_len:
            logger.warning(f"classifier token length mismatch: conll={conll_token_len} ud={ud_token_len}")
        else:
            for idx, conll_token in enumerate(conll_token_labels["tokens"]):
                if conll_token != ud_token_labels['tokens'][idx]:
                    logger.warning(f"CoNLL/UD token mismatch: idx={idx} "
                                   f"conll={conll_token} "
                                   f"ud={ud_token_labels['tokens'][idx]}")
                    is_conll_ud_aligned = False
                elif conll_token_labels["pos_tags"][idx] != ud_token_labels['xpos'][idx]:
                    logger.warning(f"CoNLL/UD part-of-speech label mismatch: token={conll_token} "
                                   f"conll={conll_token_labels['pos_tags'][idx]} "
                                   f"ud={ud_token_labels['xpos'][idx]}")

        if not ("bootstrap" in sentence_context["_"] and sentence_context["_"]["bootstrap"]):
            self.labeled_examples_conll.append(conll_token_labels)
            self.labeled_examples_ud.append(ud_token_labels)

        token_cnt = len(ud_token_labels["tokens"])
        try:
            # If bootstrap or internal, "I" and "me" are NNP.
            if (("bootstrap" in sentence_context["_"] and sentence_context["_"]["bootstrap"])
                    or ("internal" in sentence_context["_"] and sentence_context["_"]["internal"])):
                for prp_group in extract_label_idx_groups(ud_token_labels, "xpos", target_labels={"PRP"}):
                    if len(prp_group) != 1:
                        continue
                    if ud_token_labels["tokens"][prp_group[0]].lower() in {"i", "me"}:
                        ud_token_labels["xpos"][prp_group[0]] = "NNP"

            # If first token is not a proper noun and only the first character is capitalized, lower the token.
            if ("P" not in ud_token_labels["xpos"][0]
                    and ud_token_labels["tokens"][0][0].isupper()
                    and sum(ch.isupper() for ch in ud_token_labels["tokens"][0]) == 1):
                ud_token_labels["tokens"][0] = ud_token_labels["tokens"][0].lower()

            idx_to_noun_group = {}
            idx_to_noun_joined_base_form = {}
            # Extract consecutive NN* groups and split further if needed.
            for noun_group in extract_label_idx_groups(ud_token_labels, "xpos", target_labels=NOUN_LABELS):
                grp_stop_idx = noun_group[-1]
                joined_base_form = None
                for idx in noun_group:
                    if (joined_base_form is None
                            and is_plausible_noun_phrase(ud_token_labels["deprel"][idx:grp_stop_idx])):
                        # Look for joined base form from this idx to end of group
                        joined_base_form = " ".join([get_noun_base_form(
                            ud_token_labels["tokens"][idx], ud_token_labels["xpos"][idx]
                        ).lower() for idx in noun_group])
                    if joined_base_form is not None:
                        # Map each idx to group and joined form
                        idx_to_noun_group[idx] = noun_group[idx:grp_stop_idx]
                        idx_to_noun_joined_base_form[idx] = joined_base_form
                    else:
                        # Map this idx to this token's base form
                        idx_to_noun_group[idx] = [idx]
                        idx_to_noun_joined_base_form[idx] = get_noun_base_form(
                            ud_token_labels["tokens"][idx], ud_token_labels["xpos"][idx]).lower()
            # Iterate through noun groups and determine if a class or a specific thing is being referred to
            idx_to_noun_det_or_pos = {}
            for noun_group in idx_to_noun_group.values():
                dt_or_pos_idx = find_first_from_right(
                    ud_token_labels["xpos"][:noun_group[0]], {"DT", "POS", "PRP$"})
                if dt_or_pos_idx == -1:
                    next_idx = noun_group[-1]+1
                    if ud_token_labels["xpos"][next_idx] in {"IN"}:
                        dt_or_pos_idx = find_first_from_left(
                            ud_token_labels["xpos"][next_idx+1:], {"NN", "NNS", "NNP", "NNPS", "PRP$"})
                logger.info(f"noun: det_or_pos='{dt_or_pos_idx}' "
                            f"word_or_phrase='{idx_to_noun_joined_base_form[noun_group[0]]}'")
                for idx in noun_group:
                    idx_to_noun_det_or_pos[idx] = dt_or_pos_idx

        except Exception as e:
            logger.error(f"failed to process sentence periodically\n{format_exc()}")

    async def label_sentences_periodically(self):
        logger.info(f"on_periodic_run: {len(self.unlabeled_sentences)} sentences to be labeled")
        todo = []
        if self.unlabeled_sentences:
            while self.unlabeled_sentences:
                todo.append(self.unlabeled_sentences.pop(0))
        for sentence_args in todo:
            sentence, sentence_id, sentence_context = sentence_args

            emotion_labels = await get_emotions_classifications(sentence)
            ud_labels_task = asyncio.create_task(self.label_sentence(
                sentence, sentence_id, {**sentence_context, "emotions": emotion_labels["emotions"]}))
            await asyncio.gather(*[
                ud_labels_task,
            ])

    async def on_code_block_merge(self, code_block: str, code_block_id: int):
        logger.info(f"on_code_block_merge: code_block_id={code_block_id}, {code_block}")

    async def on_paragraph_merge(self, paragraph: str, paragraph_id: int, paragraph_context):
        logger.info(f"on_paragraph_merge: paragraph_id={paragraph_id}, {paragraph}")

        paragraph_soup = await run_in_threadpool(get_html_soup, f"<p>{paragraph}</p>")
        paragraph_text, paragraph_elements = await strip_html_elements(paragraph_soup, "p")
        for element in paragraph_elements:
            logger.info(f"paragraph_element: {element}")
            # TODO: If hyperlink:
            #       - Merge a domain.
            #       - Link the domain to the domain's proper noun, which will automatically connect it to the sentence.
            #       - domain name should be attributes of this domain node.
            #       - Add the protocol, path, and parameters on the vertexes to the paragraph
            #       - Add inner text as a vertex attribute

        if not paragraph_text:
            return

        previous_sentence_id = None
        async_tasks = []
        while paragraph_text:
            # Naive sentence chunking should work well enough
            sentence_end_match = naive_sentence_end_pattern.search(paragraph_text)
            if sentence_end_match:
                _, sentence_id, sentence_tasks = await self.control_plane.add_sentence(
                    paragraph_text[:sentence_end_match.end()], paragraph_context)
            else:
                _, sentence_id, sentence_tasks = await self.control_plane.add_sentence(
                    paragraph_text, paragraph_context)
            async_tasks.extend(sentence_tasks)

            async_tasks.append(
                asyncio.create_task(self.control_plane.link_paragraph_to_sentence(paragraph_id, sentence_id)))
            if previous_sentence_id is not None:
                async_tasks.append(
                    asyncio.create_task(
                        self.control_plane.link_successive_sentence(previous_sentence_id, sentence_id)))
            previous_sentence_id = sentence_id

            if sentence_end_match:
                paragraph_text = paragraph_text[sentence_end_match.end():].strip()
            else:
                paragraph_text = ""
        await asyncio.gather(*async_tasks)

    async def on_sentence_merge(self, sentence: str, sentence_id: int, sentence_context):
        if "deferred_labeling" in sentence_context["_"] and sentence_context["_"]["deferred_labeling"] is False:
            await self.label_sentence(sentence, sentence_id, sentence_context)
        else:
            # Append for deferred processing to maintain viability in memory constrained settings.
            self.unlabeled_sentences.append((sentence, sentence_id, sentence_context))


def dump_labeled_exps(exps, dir_name):
    first_exp = exps[0]
    ds_dict = {k: [] for k in first_exp.keys()}
    for exp in exps:
        for col, labels in exp.items():
            ds_dict[col].append(exp[col])
    ds = Dataset.from_dict(ds_dict)
    ts_dump = datetime.now().strftime("%Y%m%d%H%M%S")

    dump_dir = os.path.join(germ_settings.DATA_DIR, dir_name)
    os.makedirs(dump_dir, exist_ok=True)
    dump_path = f"{dump_dir}/{ts_dump}"
    logger.info(f"writing {dump_path}\n{ds}")
    ds.save_to_disk(dump_path)


def extract_consecutive_token_patterns(pos_tags, patterns):
    """
    Finds specified patterns within a list of POS tags.

    :param pos_tags: List of POS tags (e.g., ['RB', ',', 'PRP', 'VBZ', ...])
    :param patterns: List of lists, where each internal list is a series of POS tags representing a pattern
                     to search for (e.g., [['DT', 'NN'], ['IN', 'DT', 'JJ']])
    :return: Dictionary where keys are the string representation of the pattern and values are lists of
             the indexes where the pattern was found in the pos_tags list.
    """
    results = {}

    for pattern in patterns:
        pattern_length = len(pattern)
        pattern_str = '-'.join(pattern)
        results[pattern_str] = []

        # Loop through the pos_tags list and try to match each pattern
        for i in range(len(pos_tags) - pattern_length + 1):
            if pos_tags[i:i + pattern_length] == pattern:
                # Append the starting index of the pattern match to the results
                results[pattern_str].append(i)
    return results


def extract_entity_idx_groups(labels):
    entities = []
    current_entity = None
    current_indices = []

    for index, label in enumerate(labels):
        if label.startswith('B-'):  # Beginning of an entity
            # If we're in the middle of collecting an entity, store it
            if current_entity is not None:
                entities.append((current_entity, list(current_indices)))
                current_indices = []
            # Start a new entity
            current_entity = label[2:]
            current_indices.append(index)
        elif label.startswith('I-') and current_entity:  # Inside an entity
            if label[2:] == current_entity:
                current_indices.append(index)
            else:
                # If it's a continuation of a different entity, store current and reset
                entities.append((current_entity, list(current_indices)))
                current_entity = None
                current_indices = []
        else:
            # If this is an "O" or unexpected format, store current entity
            if current_entity is not None:
                entities.append((current_entity, list(current_indices)))
                current_entity = None
                current_indices = []
    # To store any entity that could end at the last element
    if current_entity is not None:
        entities.append((current_entity, current_indices))
    return entities


def extract_label_idx_groups(exp, feat, target_labels=None):
    """
    For example, given a list of labels (e.g. ["O", "O", "NN", "NN", "O", "O", "NNS", "O"]),
    this function will extract the index positions of the labels: NN, NNS, NNP, NNPS.

    It returns a list of consecutive index groupings for those noun labels.
    For example:
        ["O", "O", "NN", "NN", "O", "O", "NNS", "O"]
    would return:
        [[2, 3], [6]]

    Args:
        exp: Example
        feat: feature
        target_labels (set of str): The set of tags to target.

    Returns:
        list of lists of int: A list where each sub-list contains consecutive indices
                              of labels that match NN, NNS, NNP, NNPS.
    """
    groups = []
    current_group = []

    for idx, label in enumerate(exp[feat]):
        if (label in target_labels) if target_labels is not None else label != "O":
            # If current_group is empty or the current idx is consecutive (i.e., previous index + 1),
            # append to current_group. Otherwise, start a new group.
            if current_group and idx == current_group[-1] + 1:
                current_group.append(idx)
            else:
                if current_group:
                    groups.append(current_group)
                current_group = [idx]
        else:
            if current_group:
                groups.append(current_group)
                current_group = []

    # If there's an open group at the end, add it
    if current_group:
        groups.append(current_group)

    return groups


def extract_sentence_chunks(pos_labels):
    """
    Given a list of token POS labels (e.g., ['IN', 'DT', ... ',', ... '.']),
    return lists of consecutive index values for tokens whose POS is not punctuation.
    """
    punctuation_tags = {",", ".", ":"}
    groups = []
    current_group = []

    for i, pos_tag in enumerate(pos_labels):
        if pos_tag not in punctuation_tags:
            current_group.append(i)
        else:
            if current_group:
                # We've reached a punctuation mark, so close off the current group
                groups.append(current_group)
                current_group = []

    # If the final tokens were non-punctuation, end the last group
    if current_group:
        groups.append(current_group)

    return groups


def find_first_from_left(labels, target_labels):
    """
    Find the position of the first occurrence of any label from target_labels in the labels list.

    Args:
        labels (list of str): List of POS token labels.
        target_labels (set of str): Set of target POS labels to search for.

    Returns:
        int: The index of the first occurrence of any target label in labels.
             Returns -1 if none of the target labels are found.
    """
    for i, label in enumerate(labels):
        if label in target_labels:
            return i
    return -1


def find_first_from_right(labels, target_labels):
    """
    Returns the index of the first label (from the right)
    in 'labels' that is also in 'target_labels'.
    If no label matches, returns -1.

    :param labels: A list of POS tags (e.g. ['NNS', 'VBD', 'PRP', ...])
    :param target_labels: A list of target tags to look for
    :return: The index of the matching label or -1 if none is found
    """
    for i in range(len(labels) - 1, -1, -1):
        if labels[i] in target_labels:
            return i
    return -1


async def get_conll_token_classifications(text: str):
    async with aiohttp.ClientSession() as session:
        async with session.post("http://germ-models:9000/text/classification/conll",
                                json={"text": text}) as response:
            return await response.json()


async def get_emotions_classifications(text: str):
    async with aiohttp.ClientSession() as session:
        async with session.post("http://germ-models:9000/text/classification/emotions",
                                json={"text": text}) as response:
            return (await response.json())[0]


def get_noun_base_form(token: str, pos_label):
    if pos_label.endswith("S"):
        noun_base_form = infect_eng.singular_noun(token)
        if noun_base_form is False:
            noun_base_form = token
    else:
        noun_base_form = token
    return noun_base_form


async def get_ud_token_classifications(text: str):
    async with aiohttp.ClientSession() as session:
        async with session.post("http://germ-models:9000/text/classification/ud",
                                json={"text": text}) as response:
            return await response.json()


def idx_group_to_labels(idx_groups, all_labels):
    return [" ".join([all_labels["tokens"][i] for i in g]) for g in idx_groups]


def is_plausible_noun_phrase(deprel_labels):
    # Should not appear together in same phrase
    search_set = {'nsubj', 'nsubj:pass', 'obj', 'iobj', 'csubj', 'ccomp', 'xcomp'}
    # Intersection of input deprel_labels with the search set to find common elements
    found_elements = set(deprel_labels).intersection(search_set)
    return not (len(found_elements) > 1)


def simplify_pos_labels(pos_labels):
    new_labels = []
    for lbl in pos_labels:
        if lbl.startswith("NN"):
            lbl = "NN"
        elif lbl.startswith("VB"):
            lbl = "VB"
        new_labels.append(lbl)
    return new_labels
