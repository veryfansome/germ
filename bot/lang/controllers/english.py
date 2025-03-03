from datasets import Dataset
from datetime import datetime
from starlette.concurrency import run_in_threadpool
import aiohttp
import asyncio
import inflect
import os
import re

from bot.graph.control_plane import CodeBlockMergeEventHandler, ControlPlane, ParagraphMergeEventHandler, \
    SentenceMergeEventHandler
from bot.lang.parsers import get_html_soup, strip_html_elements
from observability.logging import logging
from settings import germ_settings

logger = logging.getLogger(__name__)

infect_eng = inflect.engine()


apostrophe_alpha_pattern = re.compile(r"'[a-z]+$")
non_word_start_or_end_pattern = re.compile(r"(^\W+|\W+$)")

non_terminal_periods = (
    r"(?<!Apt)"
    r"(?<!Blvd)"
    r"(?<!Dr)"
    r"(?<!Jr)"
    r"(?<!Mr)"
    r"(?<!Mrs)"
    r"(?<!Ms)"
    r"(?<!Ph\.D)"
    r"(?<!Rd)"
    r"(?<!Sr)"
    r"(?<!e\.g)"
    r"(?<!etc)"
    r"(?<!i\.e)"
    r"(?<![A-Z])"
)
naive_punctuation_end_pattern = re.compile(r"([,!?]\"?$|" + non_terminal_periods + r"\.\"?$)")
naive_sentence_end_pattern = re.compile(r"([\n\r]+"
                                        r"|[!?]+\"?(?=\s|$)"
                                        r"|" + non_terminal_periods + r"\.+\"?(?=\s|$))")
# Option 1:
#   [\n\r]+    - Match consecutive newline and carriage returns
# Option 2:
#   [!?]+      - Match ! or ?
#   (?=\s|$)   - Must be followed by \s or end-of-string
# Option 3:
#   non_terminal_periods  - Must not be preceded by non-terminal characters
#   \.+                   - Match .
#   (?=\s|$)              - Must be followed by \s or end-of-string


class EnglishController(CodeBlockMergeEventHandler, ParagraphMergeEventHandler, SentenceMergeEventHandler):
    def __init__(self, control_plane: ControlPlane, interval_seconds: int = 10):
        self.control_plane = control_plane
        self.dump_dir = os.path.join(germ_settings.DATA_DIR, "multi_head_exps")
        self.interval_seconds = interval_seconds
        self.labeled_multi_head_exps = []
        self.unlabeled_sentences = []

    def dump_multi_head_exps(self):
        os.makedirs(self.dump_dir, exist_ok=True)
        copied_exps = []
        if self.labeled_multi_head_exps:
            while self.labeled_multi_head_exps:
                copied_exps.append(self.labeled_multi_head_exps.pop(0))

            first_exp = copied_exps[0]
            ds_dict = {k: [] for k in first_exp.keys()}
            for exp in copied_exps:
                for col, labels in exp.items():
                    ds_dict[col].append(exp[col])
            ds = Dataset.from_dict(ds_dict)
            ts_dump = datetime.now().strftime("%Y%m%d%H%M%S")
            dump_path = f"{self.dump_dir}/{ts_dump}"
            logger.info(f"writing {dump_path}\n{ds}")
            ds.save_to_disk(dump_path)

    async def label_sentences_periodically(self):
        logger.info(f"on_periodic_run: {len(self.unlabeled_sentences)} sentences to be labeled")
        adjectives_added = set()
        nouns_added = set()

        todo = []
        if self.unlabeled_sentences:
            while self.unlabeled_sentences:
                todo.append(self.unlabeled_sentences.pop(0))
        for sentence_args in todo:
            sentence, sentence_id, sentence_context = sentence_args
            multi_head_labels = await get_token_classifications(sentence)
            self.labeled_multi_head_exps.append(multi_head_labels)
            logger.info(f"on_periodic_run: sentence_id={sentence_id}\nattrs\t{sentence_context}\n" + (
                "\n".join([f"{head}\t{labels}" for head, labels in multi_head_labels.items()])))

            # Noun groups
            idx_to_noun_group = {}
            idx_to_noun_joined_base_form = {}
            for noun_group in extract_label_idx_groups(multi_head_labels, "noun"):
                noun_group_len = len(noun_group)

                ner1_labels = []
                ner2_labels = []
                noun_labels = []
                base_form_tokens = []
                stripped_form_tokens = []
                for idx in noun_group:
                    ner1_label = multi_head_labels["ner1"][idx]
                    ner1_labels.append(ner1_label)
                    ner2_label = multi_head_labels["ner2"][idx]
                    ner2_labels.append(ner2_label)

                    noun_label = multi_head_labels["noun"][idx]
                    noun_labels.append(noun_label)

                    token = multi_head_labels["tokens"][idx]
                    stripped_token = non_word_start_or_end_pattern.sub("", token)
                    stripped_form_tokens.append(stripped_token)

                    if noun_label.endswith("S"):
                        noun_base_form = infect_eng.singular_noun(stripped_token)
                        if noun_base_form is False:
                            noun_base_form = stripped_token
                    else:
                        noun_base_form = stripped_token
                    base_form_tokens.append(noun_base_form.lower())

                # Strip any 'd, 'll, 'm, 's, etc.
                base_form_tokens[-1] = apostrophe_alpha_pattern.sub("", base_form_tokens[-1])
                stripped_form_tokens[-1] = apostrophe_alpha_pattern.sub("", stripped_form_tokens[-1])

                joined_base_form = " ".join(base_form_tokens)
                joined_stripped_form = " ".join(stripped_form_tokens)
                if joined_base_form not in nouns_added:
                    await self.control_plane.add_noun(joined_base_form)
                    await self.control_plane.add_noun_form(joined_base_form, joined_stripped_form)
                    nouns_added.add(joined_base_form)
                if ner1_labels[-1] != "O":
                    await self.control_plane.add_noun_entity_class(
                        joined_base_form, ner1_labels[-1].split("-")[-1])
                if ner2_labels[-1].endswith("LOC"):
                    await self.control_plane.add_noun_entity_class(
                        joined_base_form, ner2_labels[-1].split("-")[-1])

                # If there's a proper noun, everything becomes proper
                noun_label = "NNP" if "NNP" in noun_labels else "NN"
                # If the last noun is plural, everything becomes plural
                if noun_labels[-1].endswith("S"):
                    noun_label += "S"

                await self.control_plane.link_noun_form_to_sentence(
                    joined_base_form, joined_stripped_form,
                    noun_label,  # Implicitly last noun_label in group
                    sentence_id)

                if noun_group_len > 1:
                    for idx in range(len(base_form_tokens)):
                        ner1_label = ner1_labels[idx]
                        ner2_label = ner2_labels[idx]
                        noun_base_form = apostrophe_alpha_pattern.sub("", base_form_tokens[idx])
                        stripped_token = apostrophe_alpha_pattern.sub("", stripped_form_tokens[idx])
                        if noun_base_form not in nouns_added:
                            await self.control_plane.add_noun(noun_base_form)
                            await self.control_plane.add_noun_form(noun_base_form, stripped_token)
                            nouns_added.add(noun_base_form)
                        if ner1_label != "O":
                            await self.control_plane.add_noun_entity_class(
                                noun_base_form, ner1_label.split("-")[-1])
                        if ner2_label.endswith("LOC"):
                            await self.control_plane.add_noun_entity_class(
                                noun_base_form, ner2_label.split("-")[-1])
                        await self.control_plane.link_noun_to_phrase(noun_base_form, joined_base_form)

                for idx in noun_group:
                    idx_to_noun_group[idx] = noun_group
                    idx_to_noun_joined_base_form[idx] = joined_base_form

            # Verb groups
            idx_to_verb_group = {}
            for verb_group in extract_label_idx_groups(multi_head_labels, "verb"):
                for idx in verb_group:
                    idx_to_verb_group[idx] = verb_group
            logger.info("verb groups: %s", [f'{k}: {[multi_head_labels['tokens'][i] for i in v]}' for k, v in idx_to_verb_group.items()])

            last_adj_idx = None
            last_adj_base = None
            last_adj_stripped = None
            last_adj_token = None
            last_noun_idx = None
            last_verb = None
            last_verb_idx = None
            last_walked_idx = None
            for idx in range(len(multi_head_labels["tokens"])):
                token = multi_head_labels["tokens"][idx]
                stripped_token = non_word_start_or_end_pattern.sub("", token)
                lowered_token = stripped_token.lower()

                adj_label = multi_head_labels["adj"][idx]
                adv_label = multi_head_labels["adv"][idx]
                det_label = multi_head_labels["det"][idx]
                enc_label = multi_head_labels["enc"][idx]
                func_label = multi_head_labels["func"][idx]
                misc_label = multi_head_labels["misc"][idx]
                ner1_label = multi_head_labels["ner1"][idx]
                ner2_label = multi_head_labels["ner2"][idx]
                noun_label = multi_head_labels["noun"][idx]
                pronoun_label = multi_head_labels["pronoun"][idx]
                punct_label = multi_head_labels["punct"][idx]
                verb_label = multi_head_labels["verb"][idx]
                wh_label = multi_head_labels["wh"][idx]

                if pronoun_label == "PRP":
                    logger.info(f"{lowered_token} context: {sentence_context}")
                    # - Pronouns typically refer to the nearest preceding noun that matches the pronoun in number
                    #   and gender.
                    # - When a pronoun is used within a passage, it often retains the same referent throughout unless
                    #   otherwise specified.
                    if lowered_token == "i":
                        pass
                    elif lowered_token == "me":
                        pass
                    elif lowered_token == "you":
                        pass
                    elif lowered_token == "it":
                        pass
                    elif lowered_token == "he":
                        pass
                    elif lowered_token == "she":
                        pass
                    elif lowered_token == "they":
                        pass
                    elif lowered_token == "us":
                        pass
                    elif lowered_token == "we":
                        pass
                    elif lowered_token == "you":
                        pass
                    else:  # Other personal pronouns
                        pass

                if noun_label in {"NN", "NNS", "NNP", "NNPS"}:
                    # TODO:
                    # - Proper nouns should link to common nouns with INSTANCE_OF links
                    if (last_walked_idx is not None
                            and last_adj_base is not None
                            and last_adj_idx == last_walked_idx
                            and naive_punctuation_end_pattern.search(last_adj_token) is None):
                        # Link attributed adjectives that come immediately before nouns.
                        if last_adj_base not in adjectives_added:
                            await self.control_plane.add_adjective(last_adj_base)
                            await self.control_plane.add_adjective_form(last_adj_base, last_adj_stripped)
                            adjectives_added.add(last_adj_base)
                        await self.control_plane.link_noun_to_preceding_adjective(
                            last_adj_base, idx_to_noun_joined_base_form[idx])
                    last_noun_idx = idx

                # Filter for adjectives that are not also part of a noun phrase.
                if adj_label == "JJ" and noun_label == "O":
                    last_adj_idx = idx
                    last_adj_base = lowered_token
                    last_adj_stripped = stripped_token
                    last_adj_token = token

                if func_label == "IN":
                    # Prepositions linking nouns
                    if (last_walked_idx is not None
                            and stripped_token not in {  # Subordinating conjunctions
                                "after",
                                "although",
                                "because",
                                "before",
                                "if",
                                "once",
                                "since",
                                "that",
                                "though",
                                "unless",
                                "until",
                                "when",
                                "while",
                            }
                            and multi_head_labels["noun"][idx-1] != "O"
                            and multi_head_labels["noun"][idx+1] != "O"
                            and naive_punctuation_end_pattern.search(multi_head_labels["tokens"][idx-1]) is None):
                        await self.control_plane.link_nouns_via_preposition(
                            idx_to_noun_joined_base_form[last_walked_idx],
                            stripped_token,
                            idx_to_noun_joined_base_form[idx+1])

                if verb_label == "VB" and idx == 0:
                    # - Present tense base verb at the beginning is likely imperative sentence signal
                    # - ^ is not always the case because some constructions can start with base verbs without being
                    #   imperative in nature.
                    #   - Think what you will, our conclusions remain unchanged.
                    #   - Be it known that innovation drives progress.
                    #   - Come what may, the journey continues.
                    # - This is difficult to solve with rules because context matters. But maybe these kinds of idioms
                    #   and expressions can be learned and recalled using the graph.
                    pass
                elif verb_label == "VBZ" and idx > 0 and last_noun_idx == last_walked_idx:
                    if stripped_token == "is":
                        next_adj_label = multi_head_labels["adj"][idx + 1]
                        next_adv_label = multi_head_labels["adv"][idx + 1]
                        next_det_label = multi_head_labels["det"][idx + 1]
                        next_verb_label = multi_head_labels["verb"][idx + 1]

                # TODO:
                # - When nouns are common, verbs often describe a capability of that kind of noun
                #   - Verb nodes connect to nouns as COULD/_NOT and CAN/_NOT to indicate capability
                #   - Verb nodes should store information about associated verb links
                # - When nouns are proper, Verbs often describe actions individual do to each other
                #   - Verb links connect individual nodes

                last_walked_idx = idx

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
        # Append for deferred processing to maintain viability in memory constrained settings.
        # TODO: Make real-time graphing possible if memory is abundant.
        self.unlabeled_sentences.append((sentence, sentence_id, sentence_context))


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
        if (((label in target_labels) if target_labels is not None else label != "O")
                and (idx != 0
                     and naive_punctuation_end_pattern.search(exp["tokens"][idx-1]) is None
                     and exp["enc"][idx] == exp["enc"][idx-1])):  # Same enclosure chunk
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


async def get_token_classifications(text: str):
    async with aiohttp.ClientSession() as session:
        async with session.post("http://germ-models:9000/text/classification",
                                json={"text": text}) as response:
            return await response.json()


def idx_group_to_labels(idx_groups, all_labels):
    return [" ".join([all_labels["tokens"][i] for i in g]) for g in idx_groups]
