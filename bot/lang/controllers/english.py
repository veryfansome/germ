from datasets import Dataset
from datetime import datetime
from starlette.concurrency import run_in_threadpool
import aiohttp
import asyncio
import os
import re

from bot.graph.control_plane import CodeBlockMergeEventHandler, ControlPlane, ParagraphMergeEventHandler, SentenceMergeEventHandler
from bot.lang.parsers import get_html_soup, strip_html_elements
from observability.logging import logging
from settings import germ_settings

logger = logging.getLogger(__name__)

naive_sentence_end_pattern = re.compile(r'\S+\s+\S+([\n\r]+|[.!?]+(?=\s|$))')
# Option 1:
#   [\n\r]+    - Match consecutive newline and carriage returns
# Option 2:
#   \S+\s+\S+  - Very tolerant matching for word-space-word, which avoids matching list items like 1. or a.
#   [.!?]+     - Match . or ! or ?
#   (?=\s|$)   - Must be followed by \s or end-of-string, to avoid things like decimals and IP addresses


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
        todo = []
        if self.unlabeled_sentences:
            while self.unlabeled_sentences:
                todo.append(self.unlabeled_sentences.pop(0))
        for sentence_args in todo:
            sentence, sentence_id, sentence_parameters = sentence_args
            multi_head_labels = await get_token_classifications(sentence)
            self.labeled_multi_head_exps.append(multi_head_labels)
            logger.info(f"on_periodic_run: sentence_id={sentence_id}\nopenai={sentence_parameters}\n" + (
                "\n".join([f"{head}\t{labels}" for head, labels in multi_head_labels.items()])))

            add_positions = extract_label_idx_groups(
                multi_head_labels["misc"], {"ADD"})
            logger.info(f"add positions: {idx_group_to_labels(add_positions, multi_head_labels)}")

            cd_positions = extract_label_idx_groups(
                multi_head_labels["misc"], {"CD"})
            logger.info(f"cd positions: {idx_group_to_labels(cd_positions, multi_head_labels)}")

            det_positions = extract_label_idx_groups(
                multi_head_labels["det"], {"DT", "PDT"})
            logger.info(f"determiner positions: {idx_group_to_labels(det_positions, multi_head_labels)}")
            # DT NN .. determiner right before noun

            event_positions = extract_label_idx_groups(
                multi_head_labels["ner2"], {"B-EVENT", "I-EVENT"})
            logger.info(f"event positions: {idx_group_to_labels(event_positions, multi_head_labels)}")

            func_positions = extract_label_idx_groups(
                multi_head_labels["func"], {"CC", "IN", "RP", "TO"})
            logger.info(f"func positions: {idx_group_to_labels(func_positions, multi_head_labels)}")

            loc_positions = extract_label_idx_groups(
                multi_head_labels["ner2"], {"B-LOC", "I-LOC"})
            logger.info(f"loc positions: {idx_group_to_labels(loc_positions, multi_head_labels)}")

            noun_positions = extract_label_idx_groups(
                multi_head_labels["noun"], {"NN", "NNS", "NNP", "NNPS"})
            logger.info(f"noun positions: {idx_group_to_labels(noun_positions, multi_head_labels)}")

            org_positions = extract_label_idx_groups(
                multi_head_labels["ner1"], {"B-ORG", "I-ORG"})
            logger.info(f"org positions: {idx_group_to_labels(org_positions, multi_head_labels)}")

            person_positions = extract_label_idx_groups(
                multi_head_labels["ner1"], {"B-PER", "I-PER"})
            logger.info(f"person positions: {idx_group_to_labels(person_positions, multi_head_labels)}")

            pronoun_positions = extract_label_idx_groups(
                multi_head_labels["pronoun"], {"POS", "PRP$", "PRP"})
            logger.info(f"pronoun positions: {idx_group_to_labels(pronoun_positions, multi_head_labels)}")

            quest_positions = extract_label_idx_groups(
                multi_head_labels["punct"], {"QUESTION"})
            logger.info(f"question mark positions: {idx_group_to_labels(quest_positions, multi_head_labels)}")

            modal_positions = extract_label_idx_groups(
                multi_head_labels["verb"], {"MD"})
            logger.info(f"modal positions: {idx_group_to_labels(modal_positions, multi_head_labels)}")

            time_positions = extract_label_idx_groups(
                multi_head_labels["misc"], {"TIME"})
            logger.info(f"time positions: {idx_group_to_labels(time_positions, multi_head_labels)}")

            verb_positions = extract_label_idx_groups(
                multi_head_labels["verb"], {"VB", "VBD", "VBG", "VBN", "VBP", "VBZ"})
            logger.info(f"verb positions: {idx_group_to_labels(verb_positions, multi_head_labels)}")
            # ^VB PRP ... might be an imperative if PRP is me?

            wh_positions = extract_label_idx_groups(
                multi_head_labels["wh"], {"WDT", "WP$", "WP", "WRB"})
            logger.info(f"wh positions: {idx_group_to_labels(wh_positions, multi_head_labels)}")
            # might be interrogative

    async def on_code_block_merge(self, code_block: str, code_block_id: int):
        logger.info(f"on_code_block_merge: code_block_id={code_block_id}, {code_block}")

    async def on_paragraph_merge(self, paragraph: str, paragraph_id: int):
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
                    paragraph_text[:sentence_end_match.end()])
            else:
                _, sentence_id, sentence_tasks = await self.control_plane.add_sentence(
                    paragraph_text)
            async_tasks.extend(sentence_tasks)

            async_tasks.append(
                asyncio.create_task(self.control_plane.link_paragraph_to_sentence(paragraph_id, sentence_id)))
            if previous_sentence_id is not None:
                async_tasks.append(
                    asyncio.create_task(
                        self.control_plane.link_sentence_to_previous_sentence(previous_sentence_id, sentence_id)))
            previous_sentence_id = sentence_id

            if sentence_end_match:
                paragraph_text = paragraph_text[sentence_end_match.end():].strip()
            else:
                paragraph_text = ""
        await asyncio.gather(*async_tasks)

    async def on_sentence_merge(self, sentence: str, sentence_id: int, sentence_parameters):
        # Append for deferred processing to maintain viability in memory constrained settings.
        # TODO: Make real-time graphing possible if memory is abundant.
        self.unlabeled_sentences.append((sentence, sentence_id, sentence_parameters))


def extract_label_idx_groups(labels, target_labels):
    """
    For example, given a list of labels (e.g. ["O", "O", "NN", "NN", "O", "O", "NNS", "O"]),
    this function will extract the index positions of the labels: NN, NNS, NNP, NNPS.

    It returns a list of consecutive index groupings for those noun labels.
    For example:
        ["O", "O", "NN", "NN", "O", "O", "NNS", "O"]
    would return:
        [[2, 3], [6]]

    Args:
        labels (list of str): The list of labels.
        target_labels (set of str): The set of tags to target.

    Returns:
        list of lists of int: A list where each sub-list contains consecutive indices
                              of labels that match NN, NNS, NNP, NNPS.
    """
    groups = []
    current_group = []

    for idx, label in enumerate(labels):
        if label in target_labels:
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
