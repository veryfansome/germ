from datasets import Dataset
from datetime import datetime
from starlette.concurrency import run_in_threadpool
from traceback import format_exc
import aiohttp
import asyncio
import inflect
import os

from bot.graph.control_plane import CodeBlockMergeEventHandler, ControlPlane, ParagraphMergeEventHandler, \
    SentenceMergeEventHandler
from bot.lang.parsers import get_html_soup, strip_html_elements
from bot.lang.patterns import naive_sentence_end_pattern
from observability.logging import logging
from settings import germ_settings

logger = logging.getLogger(__name__)

infect_eng = inflect.engine()


class EnglishController(CodeBlockMergeEventHandler, ParagraphMergeEventHandler, SentenceMergeEventHandler):
    def __init__(self, control_plane: ControlPlane, interval_seconds: int = 10):
        self.control_plane = control_plane
        self.interval_seconds = interval_seconds
        self.labeled_exps_conll = []
        self.labeled_exps_ud = []
        self.unlabeled_sentences = []

    async def dump_labeled_exps(self):
        async_tasks = []
        conll_exps = []
        ud_exps = []
        if self.labeled_exps_conll:
            while self.labeled_exps_conll:
                conll_exps.append(self.labeled_exps_conll.pop(0))
            async_tasks.append(asyncio.create_task(run_in_threadpool(dump_labeled_exps, conll_exps, "conll")))
        if self.labeled_exps_ud:
            while self.labeled_exps_ud:
                ud_exps.append(self.labeled_exps_ud.pop(0))
            async_tasks.append(asyncio.create_task(run_in_threadpool(dump_labeled_exps, ud_exps, "ud")))
        await asyncio.gather(*async_tasks)

    async def graph_sentences_using_ud_model(self, sentence: str, sentence_id: int, sentence_context):
        conll_token_labels, ud_token_labels = await asyncio.gather(
            get_conll_token_classifications(sentence),
            get_ud_token_classifications(sentence),
        )
        logger.info(f"conll labels: sentence_id={sentence_id}\nattrs\t{sentence_context}\n" + (
            "\n".join([f"{head}\t{labels}" for head, labels in conll_token_labels.items()])))
        logger.info(f"ud labels: sentence_id={sentence_id}\nattrs\t{sentence_context}\n" + (
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

        self.labeled_exps_conll.append(conll_token_labels)
        self.labeled_exps_ud.append(ud_token_labels)

        adjectives_added = set()
        noun_labels = {"NN", "NNS", "NNP", "NNPS"}
        nouns_added = set()
        token_cnt = len(ud_token_labels["tokens"])
        try:
            # Noun groups / noun phrases
            idx_to_noun_group = {}
            idx_to_noun_joined_base_form = {}
            for noun_group in extract_label_idx_groups(ud_token_labels, "xpos", target_labels=noun_labels):
                noun_group_len = len(noun_group)

                base_form_tokens = []
                group_labels = []
                ner_labels = []
                raw_tokens = []
                for idx in noun_group:
                    token = ud_token_labels["tokens"][idx]
                    raw_tokens.append(token)

                    noun_label = ud_token_labels["xpos"][idx]
                    group_labels.append(noun_label)

                    if noun_label.endswith("S"):
                        noun_base_form = infect_eng.singular_noun(token)
                        if noun_base_form is False:
                            noun_base_form = token
                    else:
                        noun_base_form = token
                    base_form_tokens.append(noun_base_form.lower())

                    if is_conll_ud_aligned:
                        ner_labels.append(conll_token_labels["ner_tags"][idx])

                joined_base_form = " ".join(base_form_tokens)
                joined_raw_form = " ".join(raw_tokens)
                if joined_base_form not in nouns_added:
                    await self.control_plane.add_noun(joined_base_form)
                    await self.control_plane.add_noun_form(joined_base_form, joined_raw_form)
                    nouns_added.add(joined_base_form)
                if is_conll_ud_aligned:
                    logger.info(f"ner: {ner_labels}")

                # If there's a proper noun, everything becomes proper
                noun_label = "NNP" if "NNP" in group_labels else "NN"
                # If the last noun is plural, everything becomes plural
                if group_labels[-1].endswith("S"):
                    noun_label += "S"

                await self.control_plane.link_noun_form_to_sentence(
                    joined_base_form, joined_raw_form,
                    noun_label,  # Implicitly last noun_label in group
                    sentence_id)

                if noun_group_len > 1:
                    for idx in range(len(base_form_tokens)):
                        noun_base_form = base_form_tokens[idx]
                        raw_token = raw_tokens[idx]
                        if noun_base_form not in nouns_added:
                            await self.control_plane.add_noun(noun_base_form)
                            await self.control_plane.add_noun_form(noun_base_form, raw_token)
                            nouns_added.add(noun_base_form)
                        await self.control_plane.link_noun_to_phrase(noun_base_form, joined_base_form)

                for idx in noun_group:
                    idx_to_noun_group[idx] = noun_group
                    idx_to_noun_joined_base_form[idx] = joined_base_form

            last_adj_idx = None
            last_adj_base = None
            last_adj_token = None
            last_noun_idx = None
            #last_verb = None
            #last_verb_idx = None
            last_walked_idx = None
            for idx in range(token_cnt):
                token = ud_token_labels["tokens"][idx]
                lowered_token = token.lower()
                ud_xpos_label = ud_token_labels["xpos"][idx]

            #    if pronoun_label == "PRP":
            #        logger.info(f"{lowered_token} context: {sentence_context}")
            #        # - Pronouns typically refer to the nearest preceding noun that matches the pronoun in number
            #        #   and gender.
            #        # - When a pronoun is used within a passage, it often retains the same referent throughout unless
            #        #   otherwise specified.
            #        if lowered_token == "i":
            #            pass
            #        elif lowered_token == "me":
            #            pass
            #        elif lowered_token == "it":
            #            pass
            #        elif lowered_token == "he":
            #            pass
            #        elif lowered_token == "she":
            #            pass
            #        elif lowered_token == "they":
            #            pass
            #        elif lowered_token == "us":
            #            pass
            #        elif lowered_token == "we":
            #            pass
            #        elif lowered_token == "you":
            #            pass
            #        else:  # Other personal pronouns
            #            pass

                if ud_xpos_label in noun_labels:
                    # TODO:
                    # - Proper nouns should link to common nouns with INSTANCE_OF links
                    if (last_walked_idx is not None
                            and last_adj_base is not None
                            and last_adj_idx == last_walked_idx):
                        # Link attributed adjectives that come immediately before nouns.
                        if last_adj_base not in adjectives_added:
                            await self.control_plane.add_adjective(last_adj_base)
                            await self.control_plane.add_adjective_form(last_adj_base, last_adj_token)
                            adjectives_added.add(last_adj_base)
                        await self.control_plane.link_noun_to_preceding_adjective(
                            last_adj_base, idx_to_noun_joined_base_form[idx])
                    last_noun_idx = idx

                # Filter for adjectives that are not also part of a noun phrase.
                if ud_xpos_label == "JJ":
                    last_adj_idx = idx
                    last_adj_base = lowered_token
                    last_adj_token = token

                if ud_xpos_label == "IN":
                    # Prepositions linking nouns
                    if (last_walked_idx is not None
                            and idx < token_cnt - 1
                            and token not in {  # Subordinating conjunctions
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
                            and ud_token_labels["xpos"][idx-1] in noun_labels
                            and ud_token_labels["xpos"][idx+1] in noun_labels):
                        await self.control_plane.link_nouns_via_preposition(
                            idx_to_noun_joined_base_form[idx-1],
                            token,
                            idx_to_noun_joined_base_form[idx+1])

                # Verbs
                if ud_xpos_label in {"VB", "VBD", "VBG", "VBP", "VBN", "VBZ"}:
                    if ud_xpos_label == "VB":  # base
                        if lowered_token in {"be"}:
                            if 0 < idx < token_cnt - 1:
                                logger.info(
                                    f"state-of-being verb: {ud_token_labels['tokens'][idx - 1]} {token} {ud_token_labels['tokens'][idx + 1]}")
                    elif ud_xpos_label == "VBD":  # past tense
                        if lowered_token in {"was", "were"}:
                            if 0 < idx < token_cnt - 1:
                                logger.info(
                                    f"state-of-being verb: {ud_token_labels['tokens'][idx - 1]} {token} {ud_token_labels['tokens'][idx + 1]}")
                    elif ud_xpos_label == "VBG":  # present participle / gerund
                        if lowered_token in {"being"}:
                            if 0 < idx < token_cnt - 1:
                                logger.info(
                                    f"state-of-being verb: {ud_token_labels['tokens'][idx - 1]} {token} {ud_token_labels['tokens'][idx + 1]}")
                    elif ud_xpos_label == "VBN":  # past participle
                        if lowered_token in {"been"}:
                            if 0 < idx < token_cnt - 1:
                                logger.info(
                                    f"state-of-being verb: {ud_token_labels['tokens'][idx - 1]} {token} {ud_token_labels['tokens'][idx + 1]}")
                    elif ud_xpos_label == "VBP":  # 1st person singular present and non-3rd person singular present
                        if lowered_token in {"am", "are"}:
                            if 0 < idx < token_cnt - 1:
                                logger.info(
                                    f"state-of-being verb: {ud_token_labels['tokens'][idx - 1]} {token} {ud_token_labels['tokens'][idx + 1]}")
                    elif ud_xpos_label == "VBZ":  # 3rd person singular present
                        if lowered_token in {"is"}:
                            if idx == 0 and idx < token_cnt - 1:
                                logger.info(f"{token} {ud_token_labels['tokens'][idx+1]}")
                            elif idx < token_cnt - 1:
                                logger.info(f"{ud_token_labels['tokens'][idx-1]} {token} {ud_token_labels['tokens'][idx+1]}")

                #    # TODO:
                #    # - When nouns are common, verbs often describe a capability of that kind of noun
                #    #   - Verb nodes connect to nouns as COULD/_NOT and CAN/_NOT to indicate capability
                #    #   - Verb nodes should store information about associated verb links
                #    # - When nouns are proper, Verbs often describe actions individual do to each other
                #    #   - Verb links connect individual nodes

                last_walked_idx = idx
            pass
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
            ud_labels_task = asyncio.create_task(self.graph_sentences_using_ud_model(
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
        # Append for deferred processing to maintain viability in memory constrained settings.
        # TODO: Make real-time graphing possible if memory is abundant.
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


async def get_ud_token_classifications(text: str):
    async with aiohttp.ClientSession() as session:
        async with session.post("http://germ-models:9000/text/classification/ud",
                                json={"text": text}) as response:
            return await response.json()


def idx_group_to_labels(idx_groups, all_labels):
    return [" ".join([all_labels["tokens"][i] for i in g]) for g in idx_groups]
