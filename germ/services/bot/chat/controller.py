import aiohttp
import asyncio
import faiss
import inflect
import logging
import numpy as np
import time
from datetime import datetime
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from germ.api.models import ChatRequest, ChatResponse
from germ.data.vector_anchors import (emotion_anchors, intent_anchors, location_anchors, temporal_anchors,
                                      topic_anchors, wiki_anchors)
from germ.database.neo4j import KnowledgeGraph
from germ.observability.annotations import measure_exec_seconds
from germ.services.bot.chat.classifier import ChatRequestClassification, ChatRequestClassifier
from germ.services.bot.websocket import (WebSocketDisconnectEventHandler, WebSocketReceiveEventHandler,
                                         WebSocketSendEventHandler, WebSocketSender, WebSocketSessionMonitor)
from germ.services.models.predict.multi_predict import log_pos_labels
from germ.settings import germ_settings
from germ.utils.parsers import DocElementType, ParsedDoc, parse_markdown_doc

logger = logging.getLogger(__name__)

infect_eng = inflect.engine()


class MessageMeta(BaseModel):
    classification: ChatRequestClassification | None = None
    doc: ParsedDoc
    pos: list[dict[str, str | list[str]]]
    reconstructed_message: str
    text_embs: list[list[float]]


class ChatController(WebSocketDisconnectEventHandler, WebSocketReceiveEventHandler,
                     WebSocketSendEventHandler, WebSocketSessionMonitor):
    def __init__(
            self, knowledge_graph: KnowledgeGraph,
            delegate: WebSocketReceiveEventHandler,
    ):
        self.knowledge_graph = knowledge_graph
        self.delegate = delegate

        self.conversations: dict[int, dict] = {}
        self.faiss_emotion: faiss.IndexIDMap | None = None
        self.faiss_intent: faiss.IndexIDMap | None = None
        self.faiss_location: faiss.IndexIDMap | None = None
        self.faiss_temporal: faiss.IndexIDMap | None = None
        self.faiss_topic: faiss.IndexIDMap | None = None
        self.faiss_wiki: faiss.IndexIDMap | None = None
        self.id_to_emotion: dict[int, str] = {i: k for i, k in enumerate(emotion_anchors)}
        self.id_to_intent: dict[int, str] = {i: k for i, k in enumerate(intent_anchors)}
        self.id_to_location: dict[int, str] = {i: k for i, k in enumerate(location_anchors)}
        self.id_to_temporal: dict[int, str] = {i: k for i, k in enumerate(temporal_anchors)}
        self.id_to_topic: dict[int, str] = {i: k for i, k in enumerate(topic_anchors)}
        self.id_to_wiki: dict[int, str] = {i: k for i, k in enumerate(wiki_anchors)}
        self.sig_to_conversation_id: dict[str, set] = {}
        self.sig_to_message_meta: dict[str, MessageMeta] = {}

    async def extract_noun_groups(self, label_dict: dict[str, str | list[str]]):
        token_cnt = len(label_dict["tokens"])
        idx_to_noun_group = {}
        idx_to_noun_joined_base_form = {}
        idx_to_noun_joined_raw_form = {}
        # Extract consecutive NN* groups and split further if needed.
        for noun_group in extract_label_groups(label_dict["xpos"], target_labels={"NN", "NNS", "NNP", "NNPS"}):
            grp_stop_idx = noun_group[-1] + 1
            grp_start_idx = None
            joined_base_form = None
            joined_raw_form = None
            for idx in noun_group:
                if (joined_base_form is None
                        and is_plausible_noun_phrase(label_dict["deprel"][idx:grp_stop_idx])):
                    # Look for joined base form from this idx to end of group
                    joined_base_form = " ".join([get_noun_base_form(
                        label_dict["tokens"][idx], label_dict["xpos"][idx]
                    ).lower() for idx in noun_group])
                    joined_raw_form = " ".join([label_dict["tokens"][idx] for idx in noun_group])
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
                        label_dict["tokens"][idx], label_dict["xpos"][idx]).lower()
                    idx_to_noun_joined_raw_form[idx] = label_dict["tokens"][idx]
        # Iterate through noun groups and determine if they have an associated determiner or possessive word
        idx_to_noun_det_or_pos = {}
        logger.info(f"idx_to_noun_group: {idx_to_noun_group}")
        logger.info(f"idx_to_noun_joined_base_form: {[
            idx_to_noun_joined_base_form[g[0]] for g in idx_to_noun_group.values()
        ]}")
        for noun_group in idx_to_noun_group.values():
            # dt_or_pos_idx = find_first_from_right(
            #    label_dict["xpos"][:noun_group[0]], {"DT", "POS", "PRP$"})
            # if dt_or_pos_idx == -1 and noun_group[-1] < token_cnt - 2:
            #    next_idx = noun_group[-1] + 1
            #    if label_dict["xpos"][next_idx] in {"IN"}:
            #        dt_or_pos_idx = find_first_from_left(
            #            label_dict["xpos"][next_idx + 1:], {"NN", "NNS", "NNP", "NNPS", "PRP$"})
            # logger.info(f"noun: det_or_pos={dt_or_pos_idx} noun_group={noun_group} "
            #            f"word_or_phrase=('{idx_to_noun_joined_base_form[noun_group[0]]}', "
            #            f"'{idx_to_noun_joined_raw_form[noun_group[0]]}')")
            # for idx in noun_group:
            #    idx_to_noun_det_or_pos[idx] = dt_or_pos_idx
            pass

    async def on_disconnect(self, conversation_id: int):
        pass

    async def on_load(self):
        embedding_info = await get_text_embedding_info()
        self.faiss_emotion = faiss.IndexIDMap(faiss.IndexFlatIP(embedding_info["dim"]))
        self.faiss_intent = faiss.IndexIDMap(faiss.IndexFlatIP(embedding_info["dim"]))
        self.faiss_location = faiss.IndexIDMap(faiss.IndexFlatIP(embedding_info["dim"]))
        self.faiss_temporal = faiss.IndexIDMap(faiss.IndexFlatIP(embedding_info["dim"]))
        self.faiss_topic = faiss.IndexIDMap(faiss.IndexFlatIP(embedding_info["dim"]))
        self.faiss_wiki = faiss.IndexIDMap(faiss.IndexFlatIP(embedding_info["dim"]))

        for index, anchors, prefix in [
            (self.faiss_emotion, emotion_anchors, "emotionality: "),
            (self.faiss_intent, intent_anchors, "intent: "),
            (self.faiss_location, location_anchors, "locality: "),
            (self.faiss_temporal, temporal_anchors, "temporality: "),
            (self.faiss_topic, topic_anchors, "about: "),
            (self.faiss_wiki, wiki_anchors, "about: "),
        ]:
            anchors_len = len(anchors)
            batch_size = 1000
            for idx in range(0, anchors_len, batch_size):
                logger.info(f"getting text embedding for {min(anchors_len, batch_size)} anchors")
                embs = await get_text_embedding([prefix + a for a in anchors[idx:idx + batch_size]], prompt="passage: ")
                for emb_idx, emb in enumerate(embs):
                    await run_in_threadpool(index_embedding, index.add_with_ids, emb, idx + emb_idx)

    @measure_exec_seconds(use_logging=True, use_prometheus=True)
    async def on_receive(self, user_id: int, conversation_id: int, dt_created: datetime, text_sig: str,
                         chat_request: ChatRequest, ws_sender: WebSocketSender):
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = {
                # Indexes user_id and text_sig by dt_created
                "messages": {},
            }
        self.conversations[conversation_id]["messages"][int(dt_created.timestamp())] = {
            "user_id": user_id, "text_sig": text_sig
        }
        self.sig_to_conversation_id.get(text_sig, set()).add(conversation_id)

        # TODO: Ideas
        #   - Use text embedding signals to decide intent, topics and when to incorporate wikipedia content
        #       - Recombine several embedding signals and check similarity with original message?
        #   - Use code embedding signals to decide when to use a text browser
        #   - Use code embedding signals to decide what languages, libraries, or technologies are in play
        #   - Use POS labels to determine if the message is a question, exclamation, or statement
        #   - Use POS labels to determine if we should ask LLM to generate search queries
        #   - Train a small classifier to best combo of responses

        if text_sig not in self.sig_to_message_meta:
            classification_task = asyncio.create_task(ChatRequestClassifier.classify(chat_request))

            # Parse the message into a ParsedDoc, which includes a document scaffold (with idx pointers to text),
            # list of text blobs, and list of code blobs.
            parsed_message = await run_in_threadpool(
                parse_markdown_doc, chat_request.messages[-1].content
            )

            # Rebuild the message's text with placeholders for code blocks.
            heading_context = {}
            reconstructed_message = []
            for element_idx, scaffold_element in enumerate(parsed_message.scaffold):
                for level, heading in scaffold_element.headings.items():
                    heading_text = " ".join([parsed_message.text[idx] for idx in heading.text])
                    if level not in heading_context or heading_context[level] != heading_text:
                        heading_context[level] = heading_text
                        reconstructed_message.append(f"{'#' * level} {heading_text}")

                if scaffold_element.type == DocElementType.CODE_BLOCK:
                    # TODO: Get embeddings for code blocks as well.
                    reconstructed_message.append("\n[CODE_SNIPPET]\n")
                elif scaffold_element.type == DocElementType.LIST:
                    for item_idx, item in enumerate(scaffold_element.items):
                        reconstructed_message.append("- " + (
                            " ".join([parsed_message.text[idx] for idx in item.text])
                        ))
                elif scaffold_element.type == DocElementType.PARAGRAPH:
                    reconstructed_message.append(
                        " ".join([parsed_message.text[idx] for idx in scaffold_element.text])
                    )

            # Get embeddings and POS labels for text blobs in one go.
            reconstructed_message = "\n".join(reconstructed_message)
            text_embs, pos = await asyncio.gather(*[
                get_text_embedding([reconstructed_message]),
                get_pos_labels(parsed_message.text)
            ])

            # Cache results
            self.sig_to_message_meta[text_sig] = meta = MessageMeta(
                classification = await classification_task,
                doc=parsed_message,
                pos=pos,
                reconstructed_message=reconstructed_message,
                text_embs=text_embs
            )
        else:
            meta = self.sig_to_message_meta[text_sig]

        # Embeddings-based recall
        (
            emotion_labels, intent_labels, location_labels, temporal_labels, topic_labels, wiki_labels
        ) = await asyncio.gather(*[
            run_in_threadpool(
                search_faiss_index, self.faiss_emotion, meta.text_embs[0], self.id_to_emotion,
                num_results=3, min_sim_score=0.0
            ),
            run_in_threadpool(
                search_faiss_index, self.faiss_intent, meta.text_embs[0], self.id_to_intent,
                num_results=3, min_sim_score=0.0
            ),
            run_in_threadpool(
                search_faiss_index, self.faiss_location, meta.text_embs[0], self.id_to_location,
                num_results=3, min_sim_score=0.0
            ),
            run_in_threadpool(
                search_faiss_index, self.faiss_temporal, meta.text_embs[0], self.id_to_temporal,
                num_results=3, min_sim_score=0.15
            ),
            run_in_threadpool(
                search_faiss_index, self.faiss_topic, meta.text_embs[0], self.id_to_topic,
                num_results=3, min_sim_score=0.0
            ),
            run_in_threadpool(
                search_faiss_index, self.faiss_wiki, meta.text_embs[0], self.id_to_wiki,
                num_results=25, min_sim_score=0.0
            ),
        ])
        logger.info(f"Embedding emotion signals: {emotion_labels}")
        logger.info(f"Embedding intent signals: {intent_labels}")
        logger.info(f"Embedding location signals: {location_labels}")
        logger.info(f"Embedding temporal signals: {temporal_labels}")
        logger.info(f"Embedding topic signals: {topic_labels}")
        logger.info(f"Embedding wiki signals: {wiki_labels}")

        # Process POS labels
        for element_idx, scaffold_element in enumerate(meta.doc.scaffold):
            heading_pos = {}
            for level, heading in scaffold_element.headings.items():
                heading_pos[level] = [meta.pos[idx] for idx in heading.text]

            element_pos = []
            if scaffold_element.type == DocElementType.LIST:
                for item_idx, item in enumerate(scaffold_element.items):
                    for text_idx in item.text:
                        element_pos.append(meta.pos[text_idx])
            elif scaffold_element.type == DocElementType.PARAGRAPH:
                for text_idx in scaffold_element.text:
                    element_pos.append(meta.pos[text_idx])

            # TODO: Need to consider heading POS
            for sentence_pos in element_pos:
                #log_pos_labels(sentence_pos)
                is_exclamatory = (
                        "!" in sentence_pos["tokens"]
                        and sentence_pos["pos"][sentence_pos["tokens"].index("!")] == "PUNCT"
                )
                is_imperative = "Imp" in sentence_pos["Mood"]
                is_indicative = "Ind" in sentence_pos["Mood"]
                is_interrogative = "Int" in sentence_pos["Mood"] or (
                        "?" in sentence_pos["tokens"]
                        and sentence_pos["pos"][sentence_pos["tokens"].index("?")] == "PUNCT"
                )
                logger.info(f"Grammatical intent signals: Exc:{is_exclamatory} Imp:{is_imperative}, "
                            f"Ind:{is_indicative}, Int:{is_interrogative}")

                await self.extract_noun_groups(sentence_pos)

        # Send to LLM
        await self.delegate.on_receive(user_id, conversation_id, dt_created, text_sig, chat_request, ws_sender)

    async def on_send(self, conversation_id: int, dt_created: datetime, text_sig: str,
                      chat_response: ChatResponse, received_message_dt_created: datetime = None):
        pass

    async def on_tick(self, conversation_id: int, ws_sender: WebSocketSender):
        if conversation_id not in self.conversations:
            logger.info(f"conversation {conversation_id} is still active")
            # TODO: Handle continued conversations
            return

        timestamps = list(self.conversations[conversation_id]["messages"].keys())
        timestamps.sort()
        last_message_age_secs = int(time.time()) - timestamps.pop()
        logger.info(f"{last_message_age_secs} seconds since last message on conversation {conversation_id}")


def extract_label_groups(labels: list[str], target_labels: set[str] = None):
    """
    Given a list of labels (e.g. ["X", "X", "NN", "NN", "X", "X", "NNS", "X"]) and target labels
    (e.g. [NN, NNS, NNP, NNPS]), this function will return a list of consecutive index grouping of target labels.

    For example, given:
        ["O", "O", "NN", "NN", "O", "O", "NNS", "O"]
    this function would return:
        [[2, 3], [6]]

    Args:
        labels (list of str): Token labels
        target_labels (set of str): The set of tags to target.

    Returns:
        list of lists of int: A list where each sub-list contains consecutive indices
                              of labels that match NN, NNS, NNP, NNPS.
    """
    groups = []
    current_group = []
    for idx, label in enumerate(labels):
        if (label in target_labels) if target_labels is not None else label != "X":
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


def find_first_from_left(labels: list[str], target_labels: set[str]):
    """
    Returns the index of the first label (from the left) in 'labels' that is also in 'target_labels'.
    If no label matches, returns -1.
    """
    for i, label in enumerate(labels):
        if label in target_labels:
            return i
    return -1


def find_first_from_right(labels, target_labels):
    """
    Returns the index of the first label (from the right) in 'labels' that is also in 'target_labels'.
    If no label matches, returns -1.
    """
    for i in range(len(labels) - 1, -1, -1):
        if labels[i] in target_labels:
            return i
    return -1


async def get_pos_labels(texts: list[str]):
    async with aiohttp.ClientSession() as session:
        async with session.post(f"http://{germ_settings.MODEL_SERVICE_ENDPOINT}/text/classification/ud",
                                json={"texts": texts}) as response:
            return await response.json()


def get_noun_base_form(token: str, pos_label):
    if pos_label.endswith("S"):
        noun_base_form = infect_eng.singular_noun(token)
        if noun_base_form is False:
            noun_base_form = token
    else:
        noun_base_form = token
    return noun_base_form


async def get_text_embedding(texts: list[str], prompt: str = "query: "):
    async with aiohttp.ClientSession() as session:
        async with session.post(f"http://{germ_settings.MODEL_SERVICE_ENDPOINT}/text/embedding",
                                json={"texts": texts, "prompt": prompt}) as response:
            return (await response.json())["embeddings"]


async def get_text_embedding_info():
    async with aiohttp.ClientSession() as session:
        async with session.get(f"http://{germ_settings.MODEL_SERVICE_ENDPOINT}/text/embedding/info") as response:
            return await response.json()


def index_embedding(index_func, embedding, row_id: int):
    vector = np.array([embedding], dtype=np.float32)
    faiss.normalize_L2(vector)
    index_func(vector, np.array([row_id], dtype=np.int64))


def is_plausible_noun_phrase(deprel_labels):
    # Dependency relationship labels that should not appear together in same phrase.
    search_set = {'nsubj', 'nsubj:pass', 'obj', 'iobj', 'csubj', 'ccomp', 'xcomp'}
    # Intersection of input deprel_labels with the search set to find common elements
    found_elements = set(deprel_labels).intersection(search_set)
    return not (len(found_elements) > 1)


def search_faiss_index(index: faiss.IndexIDMap | None, embedding, id2result: dict[int, str],
                       num_results: int = 1, min_sim_score: float = 0.0):
    results = []
    if index is None:  # Be load completion
        return results

    vector = np.array([embedding], dtype=np.float32)
    faiss.normalize_L2(vector)

    sim_scores, neighbors = index.search(vector, num_results)
    for vector_id, sim_score in zip(neighbors[0], sim_scores[0]):
        if vector_id != -1 and sim_score > min_sim_score:  # -1 means no match
            results.append((id2result[vector_id], sim_score))
    return results
