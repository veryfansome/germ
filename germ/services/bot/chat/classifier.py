import aiohttp
import asyncio
import faiss
import json
import logging
import numpy as np
import re
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool
from traceback import format_exc

from germ.api.models import ChatRequest, ChatResponse
from germ.data.vector_labels import emotion_labels, intent_labels, topic_labels
from germ.database.neo4j import KnowledgeGraph
from germ.services.bot.chat import async_openai_client
from germ.settings import germ_settings
from germ.utils.parsers import ParsedDoc

logger = logging.getLogger(__name__)

summary_prefix_pattern = re.compile(r"^(The user|User):?\s*(?:\.\.\.\s*)?")


class ChatMessageClassification(BaseModel):
    foo: str | None = None


class ChatMessageMetadata(BaseModel):
    classification: ChatMessageClassification | None = None
    doc: ParsedDoc
    info_source_suggestions: list[str] | None = None
    keyword_suggestions: list[str] | None = None
    message_summary: str | None = None
    user_id: int

    @classmethod
    async def from_request(cls, user_id: int, chat_request: ChatRequest) -> "ChatMessageMetadata":
        doc_parse_task = asyncio.create_task(run_in_threadpool(ParsedDoc.from_text, chat_request.messages[-1].content))
        filtered_messages: list[dict[str, str]] = [m.model_dump() for m in chat_request.messages if m.role != "system"]

        keyword_suggestion_task = asyncio.create_task(suggest_keywords_for_search(filtered_messages))
        message_summarization_task = asyncio.create_task(summarize_most_recent_message(filtered_messages))
        doc = await doc_parse_task
        (
            keyword_suggestions,
            message_summary,
        ) = await asyncio.gather(
            keyword_suggestion_task,
            message_summarization_task,
        )
        summary_text = f"The user {message_summary['summary']}"
        logger.info(f"Message summary: {summary_text}")
        text_embs = await get_text_embedding([summary_text]),

        # TODO: Pull from Neo4j using keyword suggestions and text embedding
        info_source_candidates = [
            "arxiv.org",
            "docs.aws.amazon.com",
            "docs.cloud.google.com",
            "en.wikipedia.org",
            "en.wiktionary.org",
            "github.com",
            "huggingface.co",
            "jmlr.org",
            "registry.terraform.io",
            "stackoverflow.com",
        ]
        info_source_suggestion_task = asyncio.create_task(suggest_best_online_info_source(
            filtered_messages, info_source_candidates
        ))
        # TODO: code block summarization
        (
            info_source_suggestions,
        ) = await asyncio.gather(
            info_source_suggestion_task,
        )
        return cls(
            doc=doc,
            info_source_suggestions=info_source_suggestions["domains"],
            keyword_suggestions=keyword_suggestions["queries"],
            message_summary=message_summary["summary"],
            #text_emb=text_embs[0],
            user_id=user_id,
        )

    @classmethod
    async def from_response(cls, chat_response: ChatResponse) -> "ChatMessageMetadata":
        doc = await run_in_threadpool(
            ParsedDoc.from_text, chat_response.content
        )
        return cls(
            doc=doc,
            user_id=0
        )


class ChatRequestClassifier:
    def __init__(self, knowledge_graph: KnowledgeGraph):
        self.faiss_emotion: faiss.IndexIDMap | None = None
        self.faiss_intent: faiss.IndexIDMap | None = None
        self.faiss_topic: faiss.IndexIDMap | None = None
        self.id_to_emotion: dict[int, str] = {i: k for i, k in enumerate(emotion_labels)}
        self.id_to_intent: dict[int, str] = {i: k for i, k in enumerate(intent_labels)}
        self.id_to_topic: dict[int, str] = {i: k for i, k in enumerate(topic_labels)}
        self.knowledge_graph = knowledge_graph

    async def classify_request(self, user_id: int, chat_request: ChatRequest) -> ChatMessageMetadata:
        message_meta = await ChatMessageMetadata.from_request(user_id, chat_request)
        #await self.embedding_classifications(message_meta.text_emb)
        message_meta.classification = ChatMessageClassification()
        return message_meta

    async def classify_response(self, chat_response: ChatResponse) -> ChatMessageMetadata:
        message_meta = await ChatMessageMetadata.from_response(chat_response)
        message_meta.classification = ChatMessageClassification()
        return message_meta

    async def embedding_classifications(self, embs):
        norm_vector = await run_in_threadpool(normalize_embedding, embs)
        (
            _emotion_labels, _intent_labels, _topic_labels
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
                search_faiss_index, self.faiss_topic, norm_vector, self.id_to_topic,
                num_results=3, min_sim_score=0.0
            ),
        ])
        logger.info(f"Embedding emotion signals: {_emotion_labels}")
        logger.info(f"Embedding intent signals: {_intent_labels}")
        logger.info(f"Embedding topic signals: {_topic_labels}")

    async def dump(self):
        if self.faiss_emotion:
            faiss.write_index(self.faiss_emotion, "database_dump/faiss/faiss_emotion.index")
        if self.faiss_intent:
            faiss.write_index(self.faiss_intent, "database_dump/faiss/faiss_intent.index")
        if self.faiss_topic:
            faiss.write_index(self.faiss_topic, "database_dump/faiss/faiss_topic.index")

    async def load(self):
        embedding_info = await get_text_embedding_info()
        self.faiss_emotion = faiss.IndexIDMap(faiss.IndexFlatIP(embedding_info["dim"]))
        self.faiss_intent = faiss.IndexIDMap(faiss.IndexFlatIP(embedding_info["dim"]))
        self.faiss_topic = faiss.IndexIDMap(faiss.IndexFlatIP(embedding_info["dim"]))

        for index, labels, prefix in [
            (self.faiss_emotion, emotion_labels, "expressed emotion: "),
            (self.faiss_intent, intent_labels, "expressed intent: "),
            (self.faiss_topic, topic_labels, "about: "),
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


async def suggest_best_online_info_source(
        messages: list[dict[str, str]],
        candidates: list[str]
) -> dict[str, list[str]]:
    try:
        response = await async_openai_client.chat.completions.create(
            messages=[
                         {"role": "system",
                          "content": (
                              "You are a helpful curator of knowledge on the Internet. "
                              "Suggest the best website for finding reference information that would contribute "
                              "meaningfully to the conversation.\n"
                              "Example(s):\n" + '\n'.join(f"- {c}" for c in candidates)
                          ).strip()},
                     ] + messages,
            model="gpt-4o",
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "website",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "domains": {
                                "type": "array",
                                "description": "Website domain names.",
                                "items": {"type": "string"},
                            }
                        },
                        "required": ["domains"]
                    }
                }
            },
            n=1, timeout=30)
        suggestions = json.loads(response.choices[0].message.content)
        logger.info(f"Best info source suggestions: {suggestions}")
        assert "domains" in suggestions, "Response does not contain 'domains'"
        # TODO: Return a follow-up task based on the suggestions
        return suggestions
    except Exception:
        logger.error(f"Could not get best info source suggestions: {format_exc()}")
        return {"domains": []}


async def suggest_keywords_for_search(messages: list[dict[str, str]]) -> dict[str, list[str]]:
    try:
        response = await async_openai_client.chat.completions.create(
            messages=[
                         {"role": "system",
                          "content": (
                              "You are a helpful expert on the use of search engines. "
                              "Suggest one or more keyword queries that are likely to yield the best search "
                              "results for reference information related to the conversation. "
                          ).strip()},
                     ] + messages,
            model="gpt-4o",
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "keywords",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "queries": {
                                "type": "array",
                                "description": "Keyword queries.",
                                "items": {"type": "string"}
                            }
                        },
                        "required": ["queries"]
                    }
                }
            },
            n=1, timeout=30)
        suggestions = json.loads(response.choices[0].message.content)
        logger.info(f"Keyword suggestions: {suggestions}")
        assert "queries" in suggestions, "Response does not contain 'queries'"
        return suggestions
    except Exception:
        logger.error(f"Could not get keyword suggestions: {format_exc()}")
        return {"queries": []}


async def summarize_most_recent_message(messages: list[dict[str, str]]) -> dict[str, str]:
    try:
        response = await async_openai_client.chat.completions.create(
            messages=[
                         {"role": "system",
                          "content": (
                              "You are an efficient assistant for message summarization. "
                              "Don't respond to the user but, instead, finish the sentence \"The user ...\" by "
                              "summarizing the intent of the most recent user message as succinctly as possible. "
                          ).strip()},
                     ] + messages,
            model="gpt-4o",
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "summary",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "summary": {
                                "type": "string",
                                "description": "Message summary.",
                            }
                        },
                        "required": ["summary"]
                    }
                }
            },
            n=1, timeout=10)
        summary = json.loads(response.choices[0].message.content)
        assert "summary" in summary, "Response does not contain 'summary'"
        summary["summary"] = summary_prefix_pattern.sub("", summary["summary"]).rstrip()
        logger.info(f"Message summary: {summary}")
        return summary
    except Exception:
        logger.error(f"Could not get message summary: {format_exc()}")
        return {"summary": ""}
