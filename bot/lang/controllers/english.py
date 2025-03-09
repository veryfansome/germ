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
            self.labeled_exps_conll.append(conll_token_labels)
            self.labeled_exps_ud.append(ud_token_labels)

        adjectives_added = set()
        noun_classes_added = set()
        nouns_added = set()
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

            ##
            # Noun group extraction

            idx_to_noun_group = {}
            idx_to_noun_joined_base_form = {}
            for noun_group in extract_label_idx_groups(ud_token_labels, "xpos", target_labels=NOUN_LABELS):
                grp_start_idx = noun_group[0]
                grp_stop_idx = noun_group[-1]

                grp_base_form_tokens = []
                grp_deprel_labels = []
                grp_noun_labels = []
                grp_ner_labels = []
                grp_raw_tokens = []
                for idx in noun_group:
                    deprel_label = ud_token_labels["deprel"][idx]
                    grp_deprel_labels.append(deprel_label)

                    noun_label = ud_token_labels["xpos"][idx]
                    grp_noun_labels.append(noun_label)

                    token = ud_token_labels["tokens"][idx]
                    # If 1st token, not proper noun, and has only one upper character, convert to lower
                    if idx == 0 and "P" not in noun_label and sum(ch.isupper() for ch in token) == 1:
                        token = token.lower()

                    grp_base_form_tokens.append(get_noun_base_form(token, noun_label).lower())
                    grp_raw_tokens.append(token)

                    if is_conll_ud_aligned:
                        grp_ner_labels.append(conll_token_labels["ner_tags"][idx])

                joined_base_form = None
                joined_raw_form = None

                # The nominal subject that acts on the object shouldn't be in the same phrase entity
                if ("nsubj" in ud_token_labels["deprel"][grp_start_idx:grp_stop_idx]
                        and "obj" in ud_token_labels["deprel"][grp_start_idx:grp_stop_idx]):
                    continue

                joined_base_form = " ".join(grp_base_form_tokens)
                joined_raw_form = " ".join(grp_raw_tokens)
                #    if joined_base_form not in nouns_added:
                #        await self.control_plane.add_noun(joined_base_form, sentence_id)
                #        await self.control_plane.add_noun_form(joined_base_form, joined_raw_form, sentence_id)
                #        nouns_added.add(joined_base_form)

                #    # If there's a proper noun, everything becomes proper
                #    noun_label = "NNP" if "NNP" in grp_noun_labels else "NN"
                #    # If the last noun is plural, everything becomes plural
                #    if grp_noun_labels[-1].endswith("S"):
                #        noun_label += "S"

                #    #await self.control_plane.link_noun_form_to_sentence(
                #    #    joined_base_form, joined_raw_form,
                #    #    noun_label,
                #    #    sentence_id)
                #    if "conj" in grp_deprel_labels:
                #        await self.control_plane.link_noun_form_to_sentence(
                #            joined_base_form, joined_raw_form,
                #            "conj",
                #            sentence_id)
                #    elif "nsubj" in grp_deprel_labels:
                #        await self.control_plane.link_noun_form_to_sentence(
                #            joined_base_form, joined_raw_form,
                #            "nsubj",
                #            sentence_id)
                #    elif "obj" in grp_deprel_labels:
                #        await self.control_plane.link_noun_form_to_sentence(
                #            joined_base_form, joined_raw_form,
                #            "obj",
                #            sentence_id)

                ## If CoNLL and UD tokens are aligned, graph NER entities and phrase components, else just graph
                ## phrase components.
                #if is_conll_ud_aligned:
                #    for entity_group in extract_entity_idx_groups(grp_ner_labels):
                #        entity_label, member_idx_list = entity_group

                #        member_base_form_tokens = [grp_base_form_tokens[i] for i in member_idx_list]
                #        member_raw_form_tokens = [grp_raw_tokens[i] for i in member_idx_list]
                #        joined_member_base_form = " ".join(member_base_form_tokens)
                #        joined_member_raw_form = " ".join(member_raw_form_tokens)

                #        # Add if phrase component is new
                #        if joined_member_base_form not in nouns_added:
                #            await self.control_plane.add_noun(joined_member_base_form, sentence_id)
                #            await self.control_plane.add_noun_form(
                #                joined_member_base_form, joined_member_raw_form, sentence_id)
                #            nouns_added.add(joined_member_base_form)
                #        if entity_label != "O":
                #            logger.info(f"ner entity group: {entity_group}: {joined_member_base_form}")
                #            entity_label_lowered = entity_label.lower()
                #            if entity_label_lowered not in noun_classes_added:
                #                await self.control_plane.add_noun_class(entity_label_lowered)
                #                noun_classes_added.add(entity_label_lowered)
                #            await self.control_plane.link_noun_to_noun_class(
                #                joined_member_base_form, entity_label_lowered, sentence_id)
                #else:
                #    for idx in range(len(grp_base_form_tokens)):
                #        noun_base_form = grp_base_form_tokens[idx]
                #        raw_token = grp_raw_tokens[idx]
                #        if noun_base_form not in nouns_added:
                #            await self.control_plane.add_noun(noun_base_form, sentence_id)
                #            await self.control_plane.add_noun_form(noun_base_form, raw_token, sentence_id)
                #            nouns_added.add(noun_base_form)
                #        await self.control_plane.link_noun_to_phrase(noun_base_form, joined_base_form, sentence_id)

                for idx in noun_group:
                    token = ud_token_labels["tokens"][idx]
                    # If 1st token, not proper noun, and has only one upper character, convert to lower
                    if idx == 0 and "P" not in ud_token_labels["xpos"][idx] and sum(ch.isupper() for ch in token) == 1:
                        token = token.lower()

                    if joined_base_form is not None:
                        idx_to_noun_group[idx] = noun_group
                        idx_to_noun_joined_base_form[idx] = joined_base_form
                    else:
                        idx_to_noun_group[idx] = [idx]
                        idx_to_noun_joined_base_form[idx] = token


                    #token = ud_token_labels["tokens"][idx]

                    #grp_base_form_tokens.append(get_noun_base_form(token, noun_label).lower())
                    #grp_raw_tokens.append(token)

            ##
            # Pattern matching

            for pattern_name, start_positions in extract_consecutive_token_patterns(
                    # Simplify noun labels for pattern matching
                    simplify_pos_labels(ud_token_labels["xpos"]),
                    [
                        # Attributive adjective, noun
                        ["JJ", "NN"],
                        # Noun possesses noun
                        ["NN", "POS", "JJ", "NN"],
                        ["NN", "POS", "NN"],
                        # Noun, is/verb, a/the/the/that/.., noun
                        ["NN", "VB", "DT", "NN"],
                        # Noun, is/verb, noun
                        ["NN", "VB", "NN"],
                        # Noun, is/am/are, adjective
                        ["NN", "VB", "JJ"],
                    ]
            ).items():
                if pattern_name in {"JJ-NN"}:
                    for start_idx in start_positions:
                        jj = ud_token_labels["tokens"][start_idx]
                        jj_lowered = jj.lower()
                        if jj_lowered not in adjectives_added:
                            await self.control_plane.add_adjective(jj_lowered)
                            await self.control_plane.add_adjective_form(jj_lowered, jj)
                            adjectives_added.add(jj_lowered)
                        await self.control_plane.link_noun_to_adjective(
                            jj_lowered, idx_to_noun_joined_base_form[start_idx+1], sentence_id)
                elif pattern_name in {"NN-POS-JJ-NN"}:
                    for start_idx in start_positions:
                        await self.control_plane.link_noun_to_possessor(
                            idx_to_noun_joined_base_form[start_idx], idx_to_noun_joined_base_form[start_idx+3],
                            sentence_id)
                elif pattern_name in {"NN-POS-NN"}:
                    for start_idx in start_positions:
                        await self.control_plane.link_noun_to_possessor(
                            idx_to_noun_joined_base_form[start_idx], idx_to_noun_joined_base_form[start_idx+2],
                            sentence_id)
                elif pattern_name in {"NN-VB-NN"}:
                    for start_idx in start_positions:
                        if ud_token_labels["tokens"][start_idx+1] in {"am", "are", "is"}:
                            if ("P" in ud_token_labels["xpos"][start_idx]
                                    and "P" in ud_token_labels["xpos"][start_idx+2]):
                                # 1st noun is proper and 2nd noun is proper
                                await self.control_plane.equate_noun_to_noun(
                                    idx_to_noun_joined_base_form[start_idx], idx_to_noun_joined_base_form[start_idx+2],
                                    sentence_id)
                            elif "P" in ud_token_labels["xpos"][start_idx]:
                                pattern_cls = idx_to_noun_joined_base_form[start_idx+2]
                                if pattern_cls not in noun_classes_added:
                                    await self.control_plane.add_noun_class(pattern_cls)
                                    noun_classes_added.add(pattern_cls)
                                await self.control_plane.link_noun_to_noun_class(
                                    idx_to_noun_joined_base_form[start_idx], pattern_cls,
                                    sentence_id)
                            else:
                                pattern_cls1 = idx_to_noun_joined_base_form[start_idx]
                                if pattern_cls1 not in noun_classes_added:
                                    await self.control_plane.add_noun_class(pattern_cls1)
                                    noun_classes_added.add(pattern_cls1)
                                pattern_cls2 = idx_to_noun_joined_base_form[start_idx+2]
                                if pattern_cls2 not in noun_classes_added:
                                    await self.control_plane.add_noun_class(pattern_cls2)
                                    noun_classes_added.add(pattern_cls2)
                                await self.control_plane.link_noun_cls1_is_cls2(
                                    pattern_cls1, pattern_cls2)
                        else:
                            # TODO: Verbs also take many forms
                            # create == made == crafted, etc.
                            await self.control_plane.link_nouns_via_verb(
                                idx_to_noun_joined_base_form[start_idx],
                                wordnet_lemmatizer.lemmatize(ud_token_labels["tokens"][start_idx+1], pos="v"),
                                idx_to_noun_joined_base_form[start_idx+2],
                                sentence_id)
                elif pattern_name in {"NN-VB-DT-NN"}:
                    for start_idx in start_positions:
                        if (ud_token_labels["tokens"][start_idx+1] in {"am", "are", "is"}
                                and ud_token_labels["tokens"][start_idx+2] in {"a", "an"}):
                            pattern_noun_class = idx_to_noun_joined_base_form[start_idx+3]
                            if pattern_noun_class not in noun_classes_added:
                                await self.control_plane.add_noun_class(pattern_noun_class)
                                noun_classes_added.add(pattern_noun_class)
                            await self.control_plane.link_noun_to_noun_class(
                                idx_to_noun_joined_base_form[start_idx], pattern_noun_class,
                                sentence_id)
                        else:
                            await self.control_plane.link_nouns_via_verb(
                                idx_to_noun_joined_base_form[start_idx],
                                wordnet_lemmatizer.lemmatize(ud_token_labels["tokens"][start_idx+1], pos="v"),
                                idx_to_noun_joined_base_form[start_idx+3],
                                sentence_id)
                elif pattern_name in {"NN-VB-DT-JJ-NN"}:
                    for start_idx in start_positions:
                        if (ud_token_labels["tokens"][start_idx+1] in {"am", "are", "is"}
                                and ud_token_labels["tokens"][start_idx+2] in {"a", "an"}):
                            pattern_noun_class = idx_to_noun_joined_base_form[start_idx+4]
                            if pattern_noun_class not in noun_classes_added:
                                await self.control_plane.add_noun_class(pattern_noun_class)
                                noun_classes_added.add(pattern_noun_class)
                            await self.control_plane.link_noun_to_noun_class(
                                idx_to_noun_joined_base_form[start_idx], pattern_noun_class,
                                sentence_id)
                        else:
                            await self.control_plane.link_nouns_via_verb(
                                idx_to_noun_joined_base_form[start_idx],
                                wordnet_lemmatizer.lemmatize(ud_token_labels["tokens"][start_idx+1], pos="v"),
                                idx_to_noun_joined_base_form[start_idx+4],
                                sentence_id)
                elif pattern_name in {"NN-VB-JJ"}:
                    for start_idx in start_positions:
                        if ud_token_labels["tokens"][start_idx+1] in {"is", "am", "are"}:
                            jj = ud_token_labels["tokens"][start_idx+2]
                            jj_lowered = jj.lower()
                            if jj_lowered not in adjectives_added:
                                await self.control_plane.add_adjective(jj_lowered)
                                await self.control_plane.add_adjective_form(jj_lowered, jj)
                                adjectives_added.add(jj_lowered)
                            await self.control_plane.link_noun_to_adjective(
                                jj_lowered, idx_to_noun_joined_base_form[start_idx], sentence_id)


            ##
            # Sequence walking

            chunk_contexts = []
            for chunk_positions in extract_sentence_chunks(ud_token_labels["xpos"]):
                chunk_context = {
                    "nsubj": "",
                    "obj": "",
                }
                for idx in chunk_positions:
                    token = ud_token_labels["tokens"][idx]
                    token_lowered = token.lower()
                    ud_deprel_label = ud_token_labels["deprel"][idx]
                    ud_xpos_label = ud_token_labels["xpos"][idx]
                chunk_contexts.append(chunk_context)

               #graphed_nsubj_root_obj_link = False
                #last_adj_idx = None
                #last_adj_base = ""
                #last_adj_token = ""
                #last_noun_idx = None
                #last_nsubj_noun_idx = None
                #last_nsubj_noun = ""
                #last_obj_idx = None
                #last_obj_noun = ""
                #last_root_verb = ""
                #last_root_verb_idx = None
                #last_verb = ""
                #last_verb_idx = None
                #last_walked_idx = None
                #for idx in chunk_positions:
                #    token = ud_token_labels["tokens"][idx]
                #    lowered_token = token.lower()
                #    deprel_label = ud_token_labels["deprel"][idx]
                #    ud_xpos_label = ud_token_labels["xpos"][idx]

                #    if ud_xpos_label in NOUN_LABELS:
                #        # TODO:
                #        # - Proper nouns should link to common nouns with INSTANCE_OF links
                #        if (last_walked_idx is not None
                #                and last_adj_base
                #                and last_adj_idx == last_walked_idx):
                #            # Link attributed adjectives that come immediately before nouns.
                #            if last_adj_base not in adjectives_added:
                #                await self.control_plane.add_adjective(last_adj_base)
                #                await self.control_plane.add_adjective_form(last_adj_base, last_adj_token)
                #                adjectives_added.add(last_adj_base)
                #            await self.control_plane.link_noun_to_preceding_adjective(
                #                last_adj_base, idx_to_noun_joined_base_form[idx])
                #        if deprel_label == "nsubj":
                #            last_nsubj_noun = lowered_token
                #            last_nsubj_noun_idx = idx
                #        elif deprel_label == "obj":
                #            last_obj_noun = lowered_token
                #            last_obj_idx = idx
                #        last_noun_idx = idx

                #    elif ud_xpos_label == "JJ":
                #        last_adj_idx = idx
                #        last_adj_base = lowered_token
                #        last_adj_token = token

                #    elif ud_xpos_label == "IN":

                #        # Prepositions linking nouns
                #        if lowered_token == "concerning":
                #            lowered_token = "about"
                #        elif lowered_token == "regarding":
                #            lowered_token = "about"
                #        if lowered_token in {"about", "from", "of", "in"}:
                #            if 0 < idx:
                #                if idx < token_cnt-1:
                #                    if (ud_token_labels["xpos"][idx-1] in NOUN_LABELS
                #                            and ud_token_labels["xpos"][idx+1] in NOUN_LABELS):
                #                        # NN of NN
                #                        await self.control_plane.link_nouns_via_preposition(
                #                            idx_to_noun_joined_base_form[idx-1],
                #                            token,
                #                            idx_to_noun_joined_base_form[idx+1])
                #                elif idx < token_cnt-2:
                #                    if (ud_token_labels["xpos"][idx-1] in NOUN_LABELS
                #                            and ud_token_labels["xpos"][idx+1] in {"DT", "JJ", "PRP$"}
                #                            and ud_token_labels["xpos"][idx+2] in NOUN_LABELS):
                #                        # NN of DT/JJ/PRP$ NN
                #                        await self.control_plane.link_nouns_via_preposition(
                #                            idx_to_noun_joined_base_form[idx-1],
                #                            token,
                #                            idx_to_noun_joined_base_form[idx+2])
                #                elif idx < token_cnt-3:
                #                    if (ud_token_labels["xpos"][idx-1] in NOUN_LABELS
                #                            and ud_token_labels["xpos"][idx+1] in {"DT", "PRP$"}
                #                            and ud_token_labels["xpos"][idx+2] == "JJ"
                #                            and ud_token_labels["xpos"][idx+3] in NOUN_LABELS):
                #                        # NN of DT/PRP$ JJ NN
                #                        await self.control_plane.link_nouns_via_preposition(
                #                            idx_to_noun_joined_base_form[idx-1],
                #                            token,
                #                            idx_to_noun_joined_base_form[idx+3])
                #            elif 1 < idx:
                #                if idx < token_cnt-1:
                #                    if (ud_token_labels["xpos"][idx-2] in NOUN_LABELS
                #                            and ud_token_labels["xpos"][idx-1] in STATE_OF_BEING_VERBS
                #                            and ud_token_labels["xpos"][idx+1] in NOUN_LABELS):
                #                        # NN is of NN
                #                        await self.control_plane.link_nouns_via_preposition(
                #                            idx_to_noun_joined_base_form[idx-2],
                #                            token,
                #                            idx_to_noun_joined_base_form[idx+1])
                #                elif idx < token_cnt-2:
                #                    if (ud_token_labels["xpos"][idx-2] in NOUN_LABELS
                #                            and ud_token_labels["xpos"][idx-1] in STATE_OF_BEING_VERBS
                #                            and ud_token_labels["xpos"][idx+1] in {"DT", "JJ", "PRP$"}
                #                            and ud_token_labels["xpos"][idx+2] in NOUN_LABELS):
                #                        # NN is of DT/JJ/PRP$ NN
                #                        await self.control_plane.link_nouns_via_preposition(
                #                            idx_to_noun_joined_base_form[idx-2],
                #                            token,
                #                            idx_to_noun_joined_base_form[idx+2])
                #                elif idx < token_cnt-3:
                #                    if (ud_token_labels["xpos"][idx-2] in NOUN_LABELS
                #                            and ud_token_labels["xpos"][idx-1] in STATE_OF_BEING_VERBS
                #                            and ud_token_labels["xpos"][idx+1] in {"DT", "PRP$"}
                #                            and ud_token_labels["xpos"][idx+2] == "JJ"
                #                            and ud_token_labels["xpos"][idx+3] in NOUN_LABELS):
                #                        # NN is of DT/JJ/PRP$ NN
                #                        await self.control_plane.link_nouns_via_preposition(
                #                            idx_to_noun_joined_base_form[idx-2],
                #                            token,
                #                            idx_to_noun_joined_base_form[idx+3])

                #    # Verbs
                #    elif ud_xpos_label in VERB_LABELS:

                #        if lowered_token in STATE_OF_BEING_VERBS:
                #            if idx == 0 and idx < token_cnt - 1:
                #                logger.info(f"state-of-being verb: {token} {ud_token_labels['tokens'][idx + 1]}")
                #            elif idx < token_cnt - 1:
                #                logger.info(
                #                    f"state-of-being verb: {ud_token_labels['tokens'][idx - 1]} {token} {ud_token_labels['tokens'][idx + 1]}")

                #        if deprel_label == "root":
                #            last_root_verb = lowered_token
                #            last_root_verb_idx = idx
                #        last_verb = lowered_token
                #        last_verb_idx = idx

                #    #    # TODO:
                #    #    # - When nouns are common, verbs often describe a capability of that kind of noun
                #    #    #   - Verb nodes connect to nouns as COULD/_NOT and CAN/_NOT to indicate capability
                #    #    #   - Verb nodes should store information about associated verb links
                #    #    # - When nouns are proper, Verbs often describe actions individual do to each other
                #    #    #   - Verb links connect individual nodes

                #    if graphed_nsubj_root_obj_link is False and (last_root_verb and last_nsubj_noun and last_obj_noun):
                #        await self.control_plane.link_nouns_via_root_verb(
                #            last_nsubj_noun, wordnet_lemmatizer.lemmatize(last_root_verb, pos="v"), last_obj_noun)
                #        graphed_nsubj_root_obj_link = True

                #    last_walked_idx = idx
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


def simplify_pos_labels(pos_labels):
    new_labels = []
    for lbl in pos_labels:
        if lbl.startswith("NN"):
            lbl = "NN"
        elif lbl.startswith("VB"):
            lbl = "VB"
        new_labels.append(lbl)
    return new_labels
