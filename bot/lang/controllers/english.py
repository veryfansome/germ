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

CORE_SEMANTIC_LABELS = {
    "ccomp",  # Clausal complements with internal subjects.
    "conj",  # Word or phrase that is part of a coordination structure using "and" or "or"
    "iobj",  # The indirect object.
    "nmod",  # (nominal modifier): Indicates a noun functioning as a modifier within a larger phrase.
    "nsubj",  # (nominal subject): Marks the main subject of a clause.
    "nsubj:pass",  # A variant of nsubj capturing the subject in passive constructions.
    "obj",  # (object): Denotes the direct object of a verb.
    "obl",  # (oblique modifier): Elements that provide additional, non-core information such as location or time.
    "root",  # Main predicate or anchor of the sentence structure.
    "xcomp",  # Clausal complements with externally governed subjects
}
NOUN_LABELS = {"NN", "NNS", "NNP", "NNPS"}
VERB_LABELS = {"MD", "VB", "VBD", "VBG", "VBP", "VBN", "VBZ", "RP"}
STATE_OF_BEING_VERBS = {"am", "are", "be", "been", "being", "is", "was", "were"}
SUBORDINATING_CONJUNCTIONS = {"after", "although", "because", "before", "if", "once", "since", "that", "though",
                              "unless", "until", "when", "while"}


class EnglishController(CodeBlockMergeEventHandler, ParagraphMergeEventHandler, SentenceMergeEventHandler):
    def __init__(self, control_plane: ControlPlane, interval_seconds: int = 9):
        self.control_plane = control_plane
        self.interval_seconds = interval_seconds
        self.unlabeled_sentences = []

        self.adjectives_added = set()
        self.adverbs_added = set()
        self.noun_classes_added = set()
        self.nouns_added = set()
        self.pronouns_added = set()
        self.verbs_added = set()

    async def label_sentence(self, sentence: str, text_block_id: int, sentence_context):
        logger.info(f"sentence_context: \t{sentence_context}")

        ud_token_labels = await get_ud_token_classifications(sentence)
        token_cnt = len(ud_token_labels["tokens"])
        token_idx_positions = range(token_cnt)
        longest_token_lengths = [0 for _ in token_idx_positions]
        for idx in token_idx_positions:
            # Get longest token lengths per position
            for head in ud_token_labels.keys():
                if head == "text":
                    continue
                else:
                    label_len = len(ud_token_labels[head][idx])
                    if label_len > longest_token_lengths[idx]:
                        longest_token_lengths[idx] = label_len
        log_blobs = []
        for head, labels in ud_token_labels.items():
            # Legible formatting for examples
            if head == "text":
                log_blobs.append(f"{head}{' ' * (12-len(head))}{labels}")
                positions_blob = ''.join([
                    f"{l},{' ' * (longest_token_lengths[i] - len(str(l)) + 3)}" if i != token_cnt - 1 else str(l)
                    for i, l in enumerate(token_idx_positions)])
                log_blobs.append(f"idx{' ' * 9} {positions_blob}")
            else:
                label_blobs = []
                for idx, label in enumerate(labels):
                    label_blobs.append(f"\"{label}\",{' ' * (longest_token_lengths[idx] - len(label) + 1)}" if idx != token_cnt - 1 else f"\"{label}\"")
                    if head == "tokens":
                        continue
                    # TODO: Insert labels into Postgres

                log_blobs.append(f"{head}{' ' * (12 - len(head))}[{''.join(label_blobs)}]")
        logger.info(f"ud labels: text_block_id={text_block_id}\n" + ("\n".join(log_blobs)))

        try:
            # If first token is not a proper noun and only the first character is capitalized, lower the token.
            if ("P" not in ud_token_labels["xpos"][0]
                    and ud_token_labels["tokens"][0][0].isupper()
                    and sum(ch.isupper() for ch in ud_token_labels["tokens"][0]) == 1):
                ud_token_labels["tokens"][0] = ud_token_labels["tokens"][0].lower()

            # Extract consecutive VB* groups.
            #for verb_group in extract_label_groups(ud_token_labels, "xpos", target_labels=VERB_LABELS):
            #    logger.info([(
            #        ud_token_labels["tokens"][i],
            #        wordnet_lemmatizer.lemmatize(ud_token_labels["tokens"][i], pos="v"),
            #        ud_token_labels["xpos"][i],
            #        ud_token_labels["deprel"][i],
            #        ud_token_labels["Number"][i],
            #        ud_token_labels["Person"][i],
            #        ud_token_labels["Tense"][i],
            #        ud_token_labels["VerbForm"][i],
            #    ) for i in verb_group])

            #    grp_stop_idx = verb_group[-1] + 1
            #    grp_start_idx = verb_group[0]
            #    for idx in verb_group:
            #        token = ud_token_labels["tokens"][idx]
            #        token_lower = token.lower()
            #        token_lemma = wordnet_lemmatizer.lemmatize(token_lower, pos="v")
            #        deprel_label = ud_token_labels["deprel"][idx]
            #        if deprel_label.startswith("aux"):
            #            pass
            #        elif deprel_label == "cop":
            #            pass
            #        else:
            #            verb_cache_key = f"{token_lemma}_{text_block_id}"
            #            if verb_cache_key not in self.verbs_added:
            #                await self.control_plane.add_verb(token_lemma, text_block_id)
            #                await self.control_plane.add_verb_form(token_lemma, token, text_block_id)
            #                await self.control_plane.link_verb_form_to_sentence(
            #                    token_lemma, token,
            #                    deprel_label.replace(":", "_").upper(), text_block_id)
            #                self.verbs_added.add(verb_cache_key)

            idx_to_noun_group = {}
            idx_to_noun_joined_base_form = {}
            idx_to_noun_joined_raw_form = {}
            # Extract consecutive NN* groups and split further if needed.
            for noun_group in extract_label_groups(ud_token_labels, "xpos", target_labels=NOUN_LABELS):
                grp_stop_idx = noun_group[-1] + 1
                grp_start_idx = None
                joined_base_form = None
                joined_raw_form = None
                for idx in noun_group:
                    if (joined_base_form is None
                            and is_plausible_noun_phrase(ud_token_labels["deprel"][idx:grp_stop_idx])):
                        # Look for joined base form from this idx to end of group
                        joined_base_form = " ".join([get_noun_base_form(
                            ud_token_labels["tokens"][idx], ud_token_labels["xpos"][idx]
                        ).lower() for idx in noun_group])
                        joined_raw_form = " ".join([ud_token_labels["tokens"][idx] for idx in noun_group])
                        grp_start_idx = idx
                    if joined_base_form is not None:
                        # Map each idx to group and joined form
                        idx_to_noun_group[idx] = list(range(grp_start_idx, grp_stop_idx))
                        idx_to_noun_joined_base_form[idx] = joined_base_form
                        idx_to_noun_joined_raw_form[idx] = joined_raw_form
                    else:
                        # Map this idx to this token's base form
                        idx_to_noun_group[idx] = [idx]
                        idx_to_noun_joined_base_form[idx] = get_noun_base_form(
                            ud_token_labels["tokens"][idx], ud_token_labels["xpos"][idx]).lower()
                        idx_to_noun_joined_raw_form[idx] = ud_token_labels["tokens"][idx]
            # Iterate through noun groups and determine if they have an associated determiner or possessive word
            idx_to_noun_det_or_pos = {}
            logger.info(f"idx_to_noun_group: {idx_to_noun_group}")
            for noun_group in idx_to_noun_group.values():
                dt_or_pos_idx = find_first_from_right(
                    ud_token_labels["xpos"][:noun_group[0]], {"DT", "POS", "PRP$"})
                if dt_or_pos_idx == -1 and noun_group[-1] < token_cnt-2:
                    next_idx = noun_group[-1]+1
                    if ud_token_labels["xpos"][next_idx] in {"IN"}:
                        dt_or_pos_idx = find_first_from_left(
                            ud_token_labels["xpos"][next_idx+1:], {"NN", "NNS", "NNP", "NNPS", "PRP$"})
                logger.info(f"noun: det_or_pos={dt_or_pos_idx} noun_group={noun_group} "
                            f"word_or_phrase='{idx_to_noun_joined_base_form[noun_group[0]]}'")
                noun_cache_key = f"{idx_to_noun_joined_base_form[noun_group[0]]}_{text_block_id}"
                if noun_cache_key not in self.nouns_added:
                    await self.control_plane.add_noun(
                        idx_to_noun_joined_base_form[noun_group[0]], text_block_id)
                    await self.control_plane.add_noun_form(
                        idx_to_noun_joined_base_form[noun_group[0]], idx_to_noun_joined_raw_form[noun_group[0]],
                        text_block_id)
                    deprel_label = ud_token_labels["deprel"][noun_group[-1]]
                    await self.control_plane.link_noun_form_to_sentence(
                        idx_to_noun_joined_base_form[noun_group[0]], idx_to_noun_joined_raw_form[noun_group[0]],
                        deprel_label.replace(":", "_").upper(), text_block_id)
                    self.nouns_added.add(noun_cache_key)
                for idx in noun_group:
                    idx_to_noun_det_or_pos[idx] = dt_or_pos_idx
            #for pronoun_group in extract_label_groups(ud_token_labels, "xpos", target_labels={"PRP"}):
            #    # Assume one because pronouns don't normally appear in groups
            #    idx = pronoun_group[0]
            #    token = ud_token_labels["tokens"][idx]
            #    token_lowered = token.lower()
            #    # Rules of pronouns:
            #    # - Must agree with antecedents in number and gender when applicable
            #    # - Typically refers to most immediate noun but proximity isn't the only rule
            #    # - Reflexive pronouns refer back to the subject and need a clear and appropriate antecedent
            #    pronoun_cache_key = f"{token_lowered}_{text_block_id}"
            #    if token_lowered in {"it", "you"}:  # Subject or object
            #        pass
            #    elif token_lowered in {"i", "he", "she", "we", "they"}:  # Subject
            #        pass
            #    elif token_lowered in {"me", "him", "her", "us", "them"}:  # Object
            #        pass
            #    elif token_lowered.endswith("self") or token_lowered.endswith("selves"):  # Reflexive
            #        pass
            #    if pronoun_cache_key not in self.pronouns_added:
            #        await self.control_plane.add_pronoun(token_lowered, text_block_id)
            #        deprel_label = ud_token_labels["deprel"][pronoun_group[-1]]
            #        await self.control_plane.link_pronoun_to_sentence(
            #            token_lowered, deprel_label.replace(":", "_").upper(), text_block_id)
            #        self.pronouns_added.add(pronoun_cache_key)
            # Look for adjectives
            for pattern_name, start_positions in extract_consecutive_token_patterns(
                   # Simplify noun labels for pattern matching
                   simplify_pos_labels(ud_token_labels["xpos"]),
                   [
                       # Attributive adjective, noun
                       ["JJ", "NN"],
                       # Noun, is/are, adjective
                       ["NN", "VB", "JJ"],
                       # Noun, is/are, adverb, adjective
                       ["NN", "VB", "RB", "JJ"],
                       # Noun, is/are, either/neither, adjective, or/nor, adjective
                       ['NN', 'VB', 'CC', 'JJ', 'CC', 'JJ'],
                       # Personal pronoun, is/am/are, adjective
                       ["PRP", "VB", "JJ"],
                       # Personal pronoun, is/am/are, adverb, adjective
                       ["PRP", "VB", "RB", "JJ"],
                       # Noun, is/am/are, either/neither, adjective, or/nor, adjective
                       ['PRP', 'VB', 'CC', 'JJ', 'CC', 'JJ'],
                   ]
            ).items():
                if pattern_name in {"JJ-NN"}:
                    for start_idx in start_positions:
                        bucket = [ud_token_labels["tokens"][start_idx]]
                        # There can be multiple adjectives before a noun, sometimes separated by commas. English
                        # speakers generally follow a specific order for adjectives when they modify a noun:
                        # - Quantity or number
                        # - Quality or opinion
                        # - Size
                        # - Age
                        # - Shape
                        # - Color
                        # - Origin
                        # - Material
                        # - Purpose
                        #
                        # Example: "Three beautiful large old round red French wooden dining tables.
                        #
                        # TODO: Maybe classifying the types can be done with the multi-tagger
                        #
                        last_check_idx = start_idx
                        while last_check_idx >= 0:
                            check_idx = last_check_idx - 1
                            if ud_token_labels["xpos"][check_idx] == "JJ":
                                bucket.append(ud_token_labels["tokens"][check_idx])
                            elif ud_token_labels["xpos"][check_idx] not in {",", "CC"}:
                                break
                            last_check_idx = check_idx
                        # Currently, when multiple adjectives are grouped together, we naively attribute all adjectives
                        # to the same noun but this will mishandle some contexts. Attributes that can be reordered
                        # without changing meaning should be separated by commas but are not guaranteed to be.
                        #
                        # Examples:
                        # - "bright red car": "bright" describes "red", not "car"
                        # - "crazy loud music": "crazy" describe "loud", not "music"
                        #
                        # TODO: When adjectives are grouped together AND separated by commas, treat the groupings as
                        #       adjective phrases, e.g. "bright red".
                        #
                        for jj in bucket:
                            jj_lowered = jj.lower()
                            if jj_lowered not in self.adjectives_added:
                                await self.control_plane.add_adjective(jj_lowered)
                                await self.control_plane.add_adjective_form(jj_lowered, jj)
                                self.adjectives_added.add(jj_lowered)
                            await self.control_plane.link_adj_as_noun_attr(
                                jj_lowered, idx_to_noun_joined_base_form[start_idx + 1], text_block_id)
                elif pattern_name in {"PRP-VB-JJ", "PRP-VB-RB-JJ", "NN-VB-JJ", "NN-VB-RB-JJ"}:
                    jj_idx_shift = len(pattern_name.split("-")) - 1
                    for start_idx in start_positions:
                        if ud_token_labels["tokens"][start_idx + 1].lower() in {
                            "am", "are", "is", "’m", "'m", "’re", "'re", "’s", "'s",
                        }:
                            bucket = [ud_token_labels["tokens"][start_idx + jj_idx_shift]]
                            # There can be multiple adjectives after a linking verb
                            last_check_idx = start_idx + jj_idx_shift
                            while last_check_idx < token_cnt - 1:
                                check_idx = last_check_idx + 1
                                if ud_token_labels["xpos"][check_idx] == "JJ":
                                    bucket.append(ud_token_labels["tokens"][check_idx])
                                elif ud_token_labels["xpos"][check_idx] not in {",", "CC"}:
                                    break
                                last_check_idx = check_idx
                            negative = False
                            if (pattern_name.endswith("RB-JJ")
                                    and ud_token_labels["tokens"][start_idx + jj_idx_shift - 1] in {
                                        "not", "n’t", "n't"}):
                                negative = True
                            # Same naive attribution as above for JJ-NN
                            for jj in bucket:
                                jj_lowered = jj.lower()
                                if jj_lowered not in self.adjectives_added:
                                    await self.control_plane.add_adjective(jj_lowered)
                                    await self.control_plane.add_adjective_form(jj_lowered, jj)
                                    self.adjectives_added.add(jj_lowered)
                                if ud_token_labels["xpos"][start_idx].startswith("NN"):
                                    await self.control_plane.link_adj_as_noun_attr(
                                        jj_lowered, idx_to_noun_joined_base_form[start_idx], text_block_id,
                                        negative=negative)
                                elif ud_token_labels["xpos"][start_idx] == "PRP":
                                    await self.control_plane.link_adj_as_pronoun_to_attr(
                                        jj_lowered, ud_token_labels["tokens"][start_idx], text_block_id,
                                        negative=negative)
                elif pattern_name in {"NN-VB-CC-JJ-CC-JJ", "PRP-VB-CC-JJ-CC-JJ"}:
                    for start_idx in start_positions:
                        # TODO: Handle adjectives separated by conjunctions
                        pass
            # Look for adverbs
            for pattern_name, start_positions in extract_consecutive_token_patterns(
                    # Simplify noun labels for pattern matching
                    simplify_pos_labels(ud_token_labels["xpos"]),
                    [
                        # Adverb, adjective
                        ["RB", "JJ"],
                    ]
            ).items():
                if pattern_name in {"RB-JJ"}:
                    # TODO: A lot of adverbs are just adjectives + 'ly'
                    for start_idx in start_positions:
                        rb = ud_token_labels["tokens"][start_idx]
                        rb_lowered = rb.lower()
                        jj = ud_token_labels["tokens"][start_idx + 1]
                        jj_lowered = jj.lower()
                        if rb_lowered not in {"not", "n’t", "n't"}:
                            if rb_lowered not in self.adverbs_added:
                                await self.control_plane.add_adverb(rb_lowered)
                                if rb != rb_lowered:
                                    await self.control_plane.add_adverb_form(rb_lowered, rb)
                                self.adverbs_added.add(rb_lowered)
                            await self.control_plane.link_adv_to_adj(rb_lowered, jj_lowered)

        except Exception as e:
            logger.error(f"failed to process sentence periodically\n{format_exc()}")

    async def label_sentences_periodically(self):
        logger.info(f"on_periodic_run: {len(self.unlabeled_sentences)} sentences to be labeled")
        todo = []
        if self.unlabeled_sentences:
            while self.unlabeled_sentences:
                todo.append(self.unlabeled_sentences.pop(0))
        for sentence_args in todo:
            sentence, text_block_id, sentence_context = sentence_args

            emotion_labels = await get_emotions_classifications(sentence)
            ud_labels_task = asyncio.create_task(self.label_sentence(
                sentence, text_block_id, {**sentence_context, "emotions": emotion_labels["emotions"]}))
            await asyncio.gather(*[
                ud_labels_task,
            ])

    async def on_code_block_merge(self, code_block: str, text_block_id: int):
        logger.info(f"on_code_block_merge: text_block_id={text_block_id}, {code_block}")

    async def on_paragraph_merge(self, paragraph: str, text_block_id: int, paragraph_context):
        logger.info(f"on_paragraph_merge: text_block_id={text_block_id}, {paragraph}")

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
                asyncio.create_task(self.control_plane.link_paragraph_to_sentence(text_block_id, sentence_id)))
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

    async def on_sentence_merge(self, sentence: str, text_block_id: int, sentence_context):
        if "deferred_labeling" in sentence_context["_"] and sentence_context["_"]["deferred_labeling"] is False:
            await self.label_sentence(sentence, text_block_id, sentence_context)
        else:
            # Append for deferred processing to maintain viability in memory constrained settings.
            self.unlabeled_sentences.append((sentence, text_block_id, sentence_context))


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


def extract_entity_groups(labels):
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


def extract_label_groups(exp, feat, target_labels=None):
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
