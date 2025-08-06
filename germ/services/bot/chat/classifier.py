import aiohttp
import asyncio
import faiss
import inflect
import logging
import numpy as np
import os
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from germ.api.models import ChatRequest, ChatResponse
from germ.data.vector_labels import (emotion_labels, intent_labels, location_labels, temporal_labels,
                                      topic_labels, wiki_labels)
from germ.database.neo4j import KnowledgeGraph
from germ.services.bot.chat import async_openai_client
from germ.settings import germ_settings
from germ.utils.parsers import CodeElement, DocElementType,  ListElement, ParagraphElement, ParsedDoc

logger = logging.getLogger(__name__)

infect_eng = inflect.engine()


class ChatMessageClassification(BaseModel):
    foo: str | None = None


class ChatMessageMetadata(BaseModel):
    classification: ChatMessageClassification | None = None
    original_text: str
    sanitized_text: str | None = None
    text_emb: list[float]
    user_id: int

    @classmethod
    async def from_request(cls, user_id: int, chat_request: ChatRequest) -> "ChatMessageMetadata":

        parsed_doc, sanitized_text = await run_in_threadpool(
            ParsedDoc.from_text, chat_request.messages[-1].content
        )
        text_embs, pos = await asyncio.gather(*[
            get_text_embedding([sanitized_text]),
            get_pos_labels(parsed_doc.text)
        ])

        # TODO: Ideas
        #   - Use POS labels to determine if the message is a question, exclamation, or statement
        #   - Use POS labels to determine if we should ask LLM to generate search queries
        for element_idx, scaffold_element in enumerate(parsed_doc.scaffold):
            heading_pos, element_pos = hydrate_pos_scaffold(pos, scaffold_element)

            # TODO: Need to consider heading POS
            for sentence_pos in element_pos:
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

                extract_noun_blobs(sentence_pos)

        return cls(
            classification=ChatMessageClassification(),
            original_text=chat_request.messages[-1].content,
            sanitized_text=sanitized_text if sanitized_text != chat_request.messages[-1].content else None,
            text_emb=text_embs[0],
            user_id=user_id,
        )

    @classmethod
    async def from_response(cls, chat_response: ChatResponse) -> "ChatMessageMetadata":

        parsed_doc, sanitized_text = await run_in_threadpool(
            ParsedDoc.from_text, chat_response.content
        )
        text_embs, pos = await asyncio.gather(*[
            get_text_embedding([sanitized_text]),
            get_pos_labels(parsed_doc.text)  # TODO: Needs to be sanitized
        ])
        for element_idx, scaffold_element in enumerate(parsed_doc.scaffold):
            heading_pos, element_pos = hydrate_pos_scaffold(pos, scaffold_element)

            for sentence_pos in element_pos:
                extract_noun_blobs(sentence_pos)

        return cls(
            classification=ChatMessageClassification(),
            original_text=chat_response.content,
            sanitized_text=sanitized_text if sanitized_text != chat_response.content else None,
            text_emb=text_embs[0],
            user_id=0
        )


class ChatRequestClassifier:
    def __init__(self, knowledge_graph: KnowledgeGraph):
        self.faiss_emotion: faiss.IndexIDMap | None = None
        self.faiss_intent: faiss.IndexIDMap | None = None
        self.faiss_location: faiss.IndexIDMap | None = None
        self.faiss_temporal: faiss.IndexIDMap | None = None
        self.faiss_topic: faiss.IndexIDMap | None = None
        self.faiss_wiki: faiss.IndexIDMap | None = None
        self.id_to_emotion: dict[int, str] = {i: k for i, k in enumerate(emotion_labels)}
        self.id_to_intent: dict[int, str] = {i: k for i, k in enumerate(intent_labels)}
        self.id_to_location: dict[int, str] = {i: k for i, k in enumerate(location_labels)}
        self.id_to_temporal: dict[int, str] = {i: k for i, k in enumerate(temporal_labels)}
        self.id_to_topic: dict[int, str] = {i: k for i, k in enumerate(topic_labels)}
        self.id_to_wiki: dict[int, str] = {i: k for i, k in enumerate(wiki_labels)}
        self.knowledge_graph = knowledge_graph

    async def embedding_classifications(self, embs):
        # TODO: Ideas
        #   - Use text embedding signals to decide intent, topics and when to incorporate wikipedia content
        #       - Recombine several embedding signals and check similarity with original message?
        #   - Use code embedding signals to decide when to use a text browser
        #   - Use code embedding signals to decide what languages, libraries, or technologies are in play
        norm_vector = await run_in_threadpool(normalize_embedding, embs)
        (
            _emotion_labels, _intent_labels, _location_labels, _temporal_labels, _topic_labels, _wiki_labels
        ) = await asyncio.gather(*[
            run_in_threadpool(
                search_faiss_index, self.faiss_emotion, norm_vector, self.id_to_emotion,
                num_results=3, min_sim_score=0.0
            ),
            run_in_threadpool(
                search_faiss_index, self.faiss_intent, norm_vector, self.id_to_intent,
                num_results=3, min_sim_score=0.0
            ),
            run_in_threadpool(
                search_faiss_index, self.faiss_location, norm_vector, self.id_to_location,
                num_results=3, min_sim_score=0.1  #0.22
            ),
            run_in_threadpool(
                search_faiss_index, self.faiss_temporal, norm_vector, self.id_to_temporal,
                num_results=3, min_sim_score=0.1  #0.15
            ),
            run_in_threadpool(
                search_faiss_index, self.faiss_topic, norm_vector, self.id_to_topic,
                num_results=3, min_sim_score=0.0
            ),
            run_in_threadpool(
                search_faiss_index, self.faiss_wiki, norm_vector, self.id_to_wiki,
                num_results=25, min_sim_score=0.0
            ),
        ])
        logger.info(f"Embedding emotion signals: {_emotion_labels}")
        logger.info(f"Embedding intent signals: {_intent_labels}")
        logger.info(f"Embedding location signals: {_location_labels}")
        logger.info(f"Embedding temporal signals: {_temporal_labels}")
        logger.info(f"Embedding topic signals: {_topic_labels}")
        logger.info(f"Embedding wiki signals: {_wiki_labels}")

    async def dump(self):
        if self.faiss_emotion:
            faiss.write_index(self.faiss_emotion, "database_dump/faiss/faiss_emotion.index")
        if self.faiss_intent:
            faiss.write_index(self.faiss_intent, "database_dump/faiss/faiss_intent.index")
        if self.faiss_location:
            faiss.write_index(self.faiss_location, "database_dump/faiss/faiss_location.index")
        if self.faiss_temporal:
            faiss.write_index(self.faiss_temporal, "database_dump/faiss/faiss_temporal.index")
        if self.faiss_topic:
            faiss.write_index(self.faiss_topic, "database_dump/faiss/faiss_topic.index")
        if self.faiss_wiki:
            faiss.write_index(self.faiss_wiki, "database_dump/faiss/faiss_wiki.index")

    async def load(self):
        embedding_info = await get_text_embedding_info()
        self.faiss_emotion = faiss.IndexIDMap(faiss.IndexFlatIP(embedding_info["dim"]))
        self.faiss_intent = faiss.IndexIDMap(faiss.IndexFlatIP(embedding_info["dim"]))
        self.faiss_location = faiss.IndexIDMap(faiss.IndexFlatIP(embedding_info["dim"]))
        self.faiss_temporal = faiss.IndexIDMap(faiss.IndexFlatIP(embedding_info["dim"]))
        self.faiss_topic = faiss.IndexIDMap(faiss.IndexFlatIP(embedding_info["dim"]))
        self.faiss_wiki = faiss.IndexIDMap(faiss.IndexFlatIP(embedding_info["dim"]))

        for index, labels, prefix in [
            (self.faiss_emotion, emotion_labels, "expressed emotion: "),
            (self.faiss_intent, intent_labels, "expressed intent: "),
            (self.faiss_location, location_labels, "associated country or region: "),
            (self.faiss_temporal, temporal_labels, "associated year: "),
            (self.faiss_topic, topic_labels, "about: "),
            (self.faiss_wiki, wiki_labels, "about: "),
        ]:
            labels_len = len(labels)
            batch_size = 1000
            batch_num = 1
            for idx in range(0, labels_len, batch_size):
                logger.info(f"getting text embedding for {min(min(labels_len, batch_size) * batch_num, labels_len)}"
                            f"/{labels_len} labels")
                embs = await get_text_embedding([prefix + a for a in labels[idx:idx + batch_size]], prompt="passage: ")
                for emb_idx, emb in enumerate(embs):
                    vec = await run_in_threadpool(normalize_embedding, emb)
                    await run_in_threadpool(index.add_with_ids, vec, np.array([idx + emb_idx], dtype=np.int64))
                batch_num += 1


class LabelVectors:
    def __init__(self, name: str, dim: int, doc_prefix: str = ""):
        self.dump_file = f"database_dump/faiss/{name}_labels.index"
        self.faiss_index = (faiss.read_index(self.dump_file)
                            if os.path.exists(self.dump_file) else faiss.IndexIDMap(faiss.IndexFlatIP(dim)))
        self.doc_prefix = doc_prefix

    async def add_from_list(self, labels: list[str]):
        labels_len = len(labels)
        batch_size = 1000
        batch_num = 1
        for idx in range(0, labels_len, batch_size):
            logger.info(f"getting text embedding for {min(min(labels_len, batch_size) * batch_num, labels_len)}"
                        f"/{labels_len} labels")
            embs = await get_text_embedding([self.doc_prefix + a for a in labels[idx:idx + batch_size]], prompt="passage: ")
            for emb_idx, emb in enumerate(embs):
                vec = await run_in_threadpool(normalize_embedding, emb)
                await run_in_threadpool(self.faiss_index.add_with_ids, vec, np.array([idx + emb_idx], dtype=np.int64))
            batch_num += 1

    async def dump(self):
        await run_in_threadpool(faiss.write_index, self.faiss_index, self.dump_file)


class NounBlob(BaseModel):
    det: str | None = None
    attr: str | None = None
    blob: str


def extract_noun_blobs(label_dict: dict[str, str | list[str]]):
    token_cnt = len(label_dict["tokens"])
    idx_to_noun_group = {}
    idx_to_noun_blob = {}
    # Extract consecutive NN* groups and split further if needed.
    for noun_group in extract_pos_label_groups(label_dict["xpos"], target_labels={"NN", "NNS", "NNP", "NNPS"}):
        grp_stop_idx = noun_group[-1] + 1
        grp_start_idx = None
        noun_blob = None
        for idx in noun_group:
            if (noun_blob is None
                    and is_plausible_noun_phrase(label_dict["deprel"][idx:grp_stop_idx])):
                # Look for joined base form from this idx to end of group
                noun_blob = " ".join([label_dict["tokens"][idx] for idx in noun_group])
                grp_start_idx = idx
            if noun_blob is not None:
                # Map each idx to group and joined form
                idx_to_noun_group[idx] = list(range(grp_start_idx, grp_stop_idx))
                idx_to_noun_blob[idx] = noun_blob
            else:
                # Map this idx to this token's base form
                idx_to_noun_group[idx] = [idx]
                idx_to_noun_blob[idx] = label_dict["tokens"][idx]
    results: list[NounBlob] = []
    seen_set = set()
    for group in idx_to_noun_group.values():
        group_sig = str(group)
        if group_sig in seen_set:
            continue
        seen_set.add(group_sig)
        noun_blob = NounBlob(blob=idx_to_noun_blob[group[0]])  # Retrieve using first word idx
        results.append(noun_blob)
        # Determine if noun group have an associated determiner or possessive word
        found_idx = find_first_from_right(
            label_dict["xpos"][:group[0]], {"DT", "POS", "PRP$"})
        if found_idx != -1:
            # Found somthing on left side of noun group.
            if found_idx < group[0] - 1:
                # Check labels between found_idx and group[0].
                xpos_labels = set(label_dict["xpos"][found_idx+1:group[0]])
                if xpos_labels.issubset({"JJ", "JJR", "JJS"}):
                    # TODO: Handle comparative or superlatives
                    noun_blob.attr = " ".join(label_dict["tokens"][found_idx+1:group[0]])
                    noun_blob.det = label_dict["tokens"][found_idx]
            else:
                noun_blob.det = label_dict["tokens"][found_idx]
        #elif group[-1] < token_cnt - 2:
        #    next_idx = group[-1] + 1
        #    if label_dict["xpos"][next_idx] in {"IN"}:
        #        found_idx = find_first_from_left(
        #            label_dict["xpos"][next_idx + 1:], {"NN", "NNS", "NNP", "NNPS", "PRP$"})
        #if found_idx != -1:
        #    noun_blob.determiner = label_dict["tokens"][found_idx]
    logger.info(f"noun_blobs: {[r.model_dump(exclude_none=True) for r in results]}")


def extract_pos_label_groups(labels: list[str], target_labels: set[str] = None):
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


async def get_text_embedding(texts: list[str], prompt: str = "query: "):
    async with aiohttp.ClientSession() as session:
        async with session.post(f"http://{germ_settings.MODEL_SERVICE_ENDPOINT}/text/embedding",
                                json={"texts": texts, "prompt": prompt}) as response:
            return (await response.json())["embeddings"]


async def get_text_embedding_info():
    async with aiohttp.ClientSession() as session:
        async with session.get(f"http://{germ_settings.MODEL_SERVICE_ENDPOINT}/text/embedding/info") as response:
            return await response.json()


def hydrate_pos_scaffold(pos_labels: list[dict[str, str | list[str]]],
                         scaffold_element: CodeElement | ListElement | ParagraphElement):
    heading_pos = {}
    for level, heading in scaffold_element.headings.items():
        heading_pos[level] = [pos_labels[idx] for idx in heading.text]

    element_pos = []
    if scaffold_element.type == DocElementType.LIST:
        for item_idx, item in enumerate(scaffold_element.items):
            for text_idx in item.text:
                element_pos.append(pos_labels[text_idx])
    elif scaffold_element.type == DocElementType.PARAGRAPH:
        for text_idx in scaffold_element.text:
            element_pos.append(pos_labels[text_idx])

    return heading_pos, element_pos


def is_plausible_noun_phrase(deprel_labels):
    # Dependency relationship labels that should not appear together in same phrase.
    search_set = {'nsubj', 'nsubj:pass', 'obj', 'iobj', 'csubj', 'ccomp', 'xcomp'}
    # Intersection of input deprel_labels with the search set to find common elements
    found_elements = set(deprel_labels).intersection(search_set)
    return not (len(found_elements) > 1)


def normalize_embedding(emb: list[float]):
    vector = np.array([emb], dtype=np.float32)
    faiss.normalize_L2(vector)
    return vector


def search_faiss_index(index: faiss.IndexIDMap | None, vec, id2result: dict[int, str],
                       num_results: int = 1, min_sim_score: float = 0.0):
    results = []
    if index is None:  # Be load completion
        return results

    sim_scores, neighbors = index.search(vec, num_results)
    for vector_id, sim_score in zip(neighbors[0], sim_scores[0]):
        if vector_id != -1 and sim_score > min_sim_score:  # -1 means no match
            results.append((id2result[vector_id], sim_score))
    return results
