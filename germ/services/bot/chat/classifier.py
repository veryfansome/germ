import aiohttp
import asyncio
import faiss
import inflect
import logging
import numpy as np
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from germ.api.models import ChatRequest, ChatResponse
from germ.data.vector_labels import (emotion_labels, intent_labels, location_labels, temporal_labels,
                                      topic_labels, wiki_labels)
from germ.database.neo4j import KnowledgeGraph
from germ.services.bot.chat import async_openai_client
from germ.settings import germ_settings
from germ.utils.parsers import ParsedDoc

logger = logging.getLogger(__name__)

infect_eng = inflect.engine()


class ChatMessageClassification(BaseModel):
    foo: str | None = None


class ChatMessageMetadata(BaseModel):
    classification: ChatMessageClassification | None = None
    doc: ParsedDoc
    text_emb: list[float]
    user_id: int

    @classmethod
    async def from_request(cls, user_id: int, chat_request: ChatRequest) -> "ChatMessageMetadata":

        doc = await run_in_threadpool(
            ParsedDoc.from_text, chat_request.messages[-1].content
        )
        logger.info(doc)
        text_embs = await get_text_embedding([doc.text])

        return cls(
            classification=ChatMessageClassification(),
            doc=doc,
            text_emb=text_embs[0],
            user_id=user_id,
        )

    @classmethod
    async def from_response(cls, chat_response: ChatResponse) -> "ChatMessageMetadata":

        doc = await run_in_threadpool(
            ParsedDoc.from_text, chat_response.content
        )
        text_embs = await get_text_embedding([doc.text])

        return cls(
            classification=ChatMessageClassification(),
            doc=doc,
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
