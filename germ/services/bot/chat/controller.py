import asyncio
import json
import logging
import numpy as np
import re
import time
from datetime import datetime
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool
from traceback import format_exc
from typing import Iterable

from germ.api.models import ChatRequest, ChatResponse
from germ.database.neo4j import KnowledgeGraph
from germ.observability.annotations import measure_exec_seconds
from germ.services.bot.chat import async_openai_client
from germ.services.bot.websocket import (WebSocketDisconnectEventHandler, WebSocketReceiveEventHandler,
                                         WebSocketSendEventHandler, WebSocketSender, WebSocketSessionMonitor)
from germ.services.models import get_text_embedding
from germ.utils.parsers import ParsedDoc

logger = logging.getLogger(__name__)
message_logger = logging.getLogger('message')

summary_prefix_pattern = re.compile(r"^(The user|User):?\s*(?:\.\.\.\s*)?")


class ChatController(WebSocketDisconnectEventHandler, WebSocketReceiveEventHandler,
                     WebSocketSendEventHandler, WebSocketSessionMonitor):
    def __init__(
            self, knowledge_graph: KnowledgeGraph,
            delegate: WebSocketReceiveEventHandler,
    ):
        self.delegate = delegate
        self.conversations: dict[int, ConversationMetadata] = {}
        self.knowledge_graph = knowledge_graph

    async def on_disconnect(self, conversation_id: int):
        pass

    @measure_exec_seconds(use_logging=True, use_prometheus=True)
    async def on_receive(self, user_id: int, conversation_id: int, dt_created: datetime,
                         chat_request: ChatRequest, ws_sender: WebSocketSender):
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = ConversationMetadata(conversation_id=conversation_id)

        doc_parse_task = asyncio.create_task(
            run_in_threadpool(ParsedDoc.from_text, chat_request.messages[-1].content)
        )
        # TODO: code block summarization

        # Generate embeddings on summarized message
        filtered_messages: list[dict[str, str]] = [m.model_dump() for m in chat_request.messages if m.role != "system"]
        summary_text = f"The user {(await summarize_most_recent_message(filtered_messages))['summary']}"
        logger.info(f"Message summary: {summary_text}")
        text_emb = await run_in_threadpool(
            normalize_embedding, np.array((await get_text_embedding([summary_text]))[0], dtype=np.float32)
        )
        text_emb_floats = text_emb.tolist()

        # Recall
        (
            most_similar_search_queries,
        ) = await asyncio.gather(
            self.knowledge_graph.match_search_queries_by_similarity(text_emb_floats, k=5),
        )
        logger.info(f"Recalled search queries: {[(r['score'], r['text']) for r in most_similar_search_queries]}")
        most_similar_search_queries = {r["text"]: r["embedding"] for r in most_similar_search_queries}

        # TODO: Pull from Neo4j using search_query suggestions and text embedding
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
        (
            info_source_suggestions,
            search_query_suggestions
        ) = await asyncio.gather(
            suggest_best_online_info_source(filtered_messages, info_source_candidates),
            suggest_search_query(filtered_messages, most_similar_search_queries.keys())
        )

        doc = await doc_parse_task
        message_meta = ChatMessageMetadata(
            doc=doc,
            info_source_suggestions=info_source_suggestions["domains"],
            search_query_suggestions=search_query_suggestions["queries"],
            message_summary=summary_text,
            user_id=user_id,
        )

        await asyncio.gather(
            self.conversations[conversation_id].add_message(dt_created, message_meta, log=True),
            self.delegate.on_receive(user_id, conversation_id, dt_created, chat_request, ws_sender),  # Send to LLM
            self.update_search_query_embeddings(
                conversation_id, dt_created,
                text_emb, most_similar_search_queries, search_query_suggestions["queries"]
            )
        )

    async def on_send(self, conversation_id: int, dt_created: datetime,
                      chat_response: ChatResponse, received_message_dt_created: datetime = None):
        if conversation_id not in self.conversations:
            logger.error(f"Conversation {conversation_id} not found.")
            return

        doc = await run_in_threadpool(
            ParsedDoc.from_text, chat_response.content
        )
        message_meta = ChatMessageMetadata(
            doc=doc,
            user_id=0
        )
        await self.conversations[conversation_id].add_message(dt_created, message_meta, log=True)

    async def on_start(self):
        pass

    async def on_tick(self, conversation_id: int, ws_sender: WebSocketSender):
        if conversation_id not in self.conversations:
            logger.error(f"Conversation {conversation_id} not found.")
            return

        timestamps = list(self.conversations[conversation_id].messages.keys())
        timestamps.sort()
        last_message_age_secs = int(time.time()) - timestamps.pop().timestamp()
        logger.info(f"{last_message_age_secs} seconds since last message on conversation {conversation_id}")
        # TODO:
        #   - If a conversation has no user messages and has been idle for some time, recall what you know about
        #     the user and attempt to start a conversation.


    async def update_search_query_embeddings(
            self, conversation_id, dt_created: datetime,
            new_embedding: np.ndarray, recalled_embeddings: dict[str, list[float]], new_suggestions: list[str]
    ):
        # Check if suggested search queries are known but were missed during earlier recall
        matched_suggestions = await self.knowledge_graph.match_search_queries_by_text([
            q for q in new_suggestions if q not in recalled_embeddings
        ])
        # Drop recalled search queries that were not selected
        recalled_embeddings = {txt: emb for txt, emb in recalled_embeddings.items() if txt in new_suggestions}
        recalled_embeddings.update(matched_suggestions)

        embeddings_to_update: dict[str, list[float]] = {}
        # For each recalled embedding, average the new and the recalled embedding to form a new centroid
        for text, recalled_embedding in recalled_embeddings.items():
            embeddings_to_update[text] = (
                    (new_embedding + np.array(recalled_embedding, dtype=np.float32)) / 2
            ).tolist()
        # Use the new embedding for anything that is completely new.
        for text in new_suggestions:
            if text not in recalled_embeddings:
                embeddings_to_update[text] = new_embedding.tolist()

        await self.knowledge_graph.add_search_queries(
            conversation_id, dt_created, embeddings_to_update
        )


class ChatMessageMetadata(BaseModel):
    doc: ParsedDoc
    info_source_suggestions: list[str] | None = None
    message_summary: str | None = None
    search_query_suggestions: list[str] | None = None
    user_id: int


class ConversationMetadata(BaseModel):
    conversation_id: int
    messages: dict[datetime, ChatMessageMetadata] = {}

    async def add_message(self, dt_created: datetime, message_meta: ChatMessageMetadata, log: bool = False):
        self.messages[dt_created] = message_meta
        if log:
            message_logger.info(
                json.dumps(
                    [int(dt_created.timestamp()), self.conversation_id, message_meta.model_dump(exclude_none=True)],
                    separators=(",", ":")
                )
            )


def normalize_embedding(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm == 0:  # avoid division by zero
        return vec
    return vec / norm


async def suggest_best_online_info_source(
        messages: list[dict[str, str]],
        candidates: list[str]
) -> dict[str, list[str]]:
    try:
        response = await async_openai_client.chat.completions.create(
            messages=[
                         {"role": "system",
                          "content": (
                                  "You are a helpful curator of information on the Internet. "
                                  
                                  "Suggest the best website for finding something that would contribute meaningfully "
                                  "to the conversation. "
                                  
                                  "Limit your suggestions to websites that work well with text-based browsers. "
                                  
                                  "\nExample(s):\n" + '\n'.join(f"- {c}" for c in candidates)
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


async def suggest_search_query(
        messages: list[dict[str, str]],
        similar_candidates: Iterable[str],
) -> dict[str, list[str]]:
    prompt = (
        "You are a helpful expert on the use of search engines. "

        "Suggest one or more keyword queries that are likely to yield the best search "
        "results related to the conversation. "
    ).strip()
    if similar_candidates:
        prompt += (
            "\nConsider the following candidate(s):\n" + '\n'.join(f"- {c}" for c in similar_candidates)
        )
    try:
        response = await async_openai_client.chat.completions.create(
            messages=[{"role": "system", "content": prompt}] + messages,
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
        logger.info(f"Search query suggestions: {suggestions}")
        assert "queries" in suggestions, "Response does not contain 'queries'"
        return suggestions
    except Exception:
        logger.error(f"Could not get search query suggestions: {format_exc()}")
        return {"queries": []}


async def summarize_most_recent_message(messages: list[dict[str, str]]) -> dict[str, str]:
    summary: str | None = None
    while not summary:
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
            response_content = json.loads(response.choices[0].message.content)
            assert "summary" in response_content, "Response does not contain 'summary'"
            summary = summary_prefix_pattern.sub("", response_content["summary"]).rstrip()
        except Exception:
            logger.error(f"Could not get message summary: {format_exc()}")
    return {"summary": summary}
