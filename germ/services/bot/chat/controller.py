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
from germ.services.bot.chat import async_openai_client, openai_handlers
from germ.services.bot.chat.classifier import UserIntentClassifier
from germ.services.bot.websocket import (WebSocketDisconnectEventHandler, WebSocketReceiveEventHandler,
                                         WebSocketSendEventHandler, WebSocketSender, WebSocketSessionMonitor)
from germ.services.models import get_text_embedding
from germ.settings import germ_settings
from germ.utils.parsers import ParsedDoc

logger = logging.getLogger(__name__)
message_logger = logging.getLogger('message')

summary_prefix_pattern = re.compile(r"^(The user|User):?\s*(?:\.\.\.\s*)?")


class ChatController(WebSocketDisconnectEventHandler, WebSocketReceiveEventHandler,
                     WebSocketSendEventHandler, WebSocketSessionMonitor):
    def __init__(self, knowledge_graph: KnowledgeGraph):
        self.conversations: dict[int, ConversationMetadata] = {}
        self.knowledge_graph = knowledge_graph

    async def on_disconnect(self, conversation_id: int):
        pass

    @measure_exec_seconds(use_logging=True, use_prometheus=True)
    async def on_receive(self, user_id: int, conversation_id: int, dt_created: datetime,
                         chat_request: ChatRequest, ws_sender: WebSocketSender):
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = ConversationMetadata(conversation_id=conversation_id)

        if chat_request.uploaded_filenames:
            for filename in chat_request.uploaded_filenames:
                # TODO: handle uploaded files
                pass

        doc_parse_task = asyncio.create_task(
            run_in_threadpool(ParsedDoc.from_text, chat_request.messages[-1].content)
        )
        # TODO: code block summarization

        filtered_messages: list[dict[str, str]] = [m.model_dump() for m in chat_request.messages if m.role != "system"]

        (
            message_summary,
            user_intent_labels,
        ) = await asyncio.gather(
            summarize_message_received(filtered_messages),
            UserIntentClassifier.suggest_user_intent(filtered_messages),
        )
        summary_text = f"The user {message_summary['summary']}"
        logger.info(f"User message summary:\n - {summary_text}")
        logger.info(f"User intent labels: {''.join(f'\n - {l}' for l in user_intent_labels['intents'])}")

        # Generate embeddings on summarized user message
        summary_emb = await run_in_threadpool(
            normalize_embeddings, np.array(await get_text_embedding([summary_text]), dtype=np.float32)
        )
        summary_emb_floats = summary_emb[0].tolist()

        # TODO:
        #   - handle if the user is disagreeing with something I said previously
        #   - handle if the user is simply offering agreement or a compliment
        #   - handle if situations that don't require information search

        # Recall
        (
            recalled_bot_message_summaries,
            recalled_user_message_summaries,
        ) = await asyncio.gather(
            self.knowledge_graph.match_bot_message_summaries_by_similarity_to_message_received(
                conversation_id, summary_emb_floats, k=15, min_similarity=0.85,
            ),
            self.knowledge_graph.match_user_message_summaries_by_similarity_to_message_received(
                conversation_id, user_id, summary_emb_floats, k=15, min_similarity=0.85,
            ),
        )
        recalled_search_queries = await self.knowledge_graph.match_search_queries_by_similarity_to_message_received(
            summary_emb_floats, [
                {
                    "conversation_id": struct["conversation_id"],
                    "dt_created": struct["dt_created"],
                    "score": struct["score"],
                } for struct in recalled_user_message_summaries
            ], alpha=0.7, k=5,
        )
        # Filter out recalled summaries from the current conversation, they used for search query recall but not after.
        recalled_bot_message_summaries = [s for s in recalled_bot_message_summaries
                                          if s["conversation_id"] != conversation_id]
        recalled_user_message_summaries = [s for s in recalled_user_message_summaries
                                          if s["conversation_id"] != conversation_id]
        logger.info(f"Recalled bot message summaries: {''.join(
            ('\n - ' + str((r['score'], r['text'], r['conversation_id'], r['dt_created'])))
            for r in recalled_bot_message_summaries
        )}")
        logger.info(f"Recalled user message summaries: {''.join(
            ("\n - " + str((r['score'], r['text'], r['conversation_id'], r['dt_created'])))
            for r in recalled_user_message_summaries
        )}")
        logger.info(f"Recalled search queries: {''.join(
            ("\n - " + str((r['score'], r['text']))) for r in recalled_search_queries
        )}")

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
            suggest_search_query(filtered_messages, [r["text"] for r in recalled_search_queries])
        )
        logger.info(f"Search query suggestions: {search_query_suggestions['queries']}")

        doc = await doc_parse_task
        message_meta = ChatMessageMetadata(
            doc=doc,
            info_source_suggestions=info_source_suggestions["domains"],
            search_query_suggestions=search_query_suggestions["queries"],
            message_summary=summary_text,
            user_id=user_id,
        )

        if germ_settings.REASONING_MODEL:
            delegate_handler = openai_handlers.ReasoningChatModelEventHandler()
        else:
            delegate_handler = openai_handlers.ChatModelEventHandler()

        await asyncio.gather(
            # Complete chat request
            delegate_handler.on_receive(user_id, conversation_id, dt_created, chat_request, ws_sender),

            # House-keeping
            self.conversations[conversation_id].add_message(dt_created, message_meta, log=True),
            self.knowledge_graph.add_summary(conversation_id, dt_created, {
                summary_text: (0, summary_emb_floats)
            }),
            self.knowledge_graph.add_user_message_intent(conversation_id, dt_created, [
                l.lower().split(": ") for l in user_intent_labels['intents']
            ]),
            self.update_search_query_embeddings(conversation_id, dt_created, search_query_suggestions["queries"])
        )

    async def on_send(self, conversation_id: int, dt_created: datetime,
                      chat_response: ChatResponse, received_message_dt_created: datetime = None):
        if conversation_id not in self.conversations:
            logger.error(f"Conversation {conversation_id} not found.")
            return

        doc_parse_task = asyncio.create_task(
            run_in_threadpool(ParsedDoc.from_text, chat_response.content)
        )

        # Hydrate message history from conversation metadata
        messages = await run_in_threadpool(self.conversations[conversation_id].hydrate_messages)
        messages.append({"role": "assistant", "content": chat_response.content})

        # Generate embeddings on summarized assistant response
        summary_statements = (await summarize_message_sent(messages))["statements"]
        logger.info(f"Response summary: {''.join(
            f'\n - {s}' for s in summary_statements
        )}")
        text_embs = await run_in_threadpool(
            normalize_embeddings, np.array(await get_text_embedding(summary_statements), dtype=np.float32)
        )

        doc = await doc_parse_task
        message_meta = ChatMessageMetadata(
            doc=doc,
            user_id=0
        )

        await asyncio.gather(
            self.conversations[conversation_id].add_message(dt_created, message_meta, log=True),
            # TODO: Check similarity and dedupe
            self.knowledge_graph.add_summary(conversation_id, dt_created, {
                txt: (idx, text_embs[idx].tolist()) for idx, txt in enumerate(summary_statements)
            })
        )

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
            self, conversation_id, dt_created: datetime, new_suggestions: list[str]
    ):
        new_suggestions_embedding = await run_in_threadpool(
            normalize_embeddings, np.array(await get_text_embedding(new_suggestions), dtype=np.float32)
        )
        await self.knowledge_graph.add_search_queries(
            conversation_id, dt_created, {
                txt: new_suggestions_embedding[idx].tolist() for idx, txt in enumerate(new_suggestions)
            }
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

    def hydrate_messages(self) -> list[dict[str, str]]:
        return [
            {
                "role": ("assistant" if meta.user_id == 0 else "user"),
                "content": meta.doc.restore()
            }
            for _, meta in self.messages.items()
        ]


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Prevent division by zero by setting zero norms to one
    return embeddings / norms


async def suggest_best_online_info_source(
        messages: list[dict[str, str]],
        candidates: list[str]
) -> dict[str, list[str]]:
    prompt = (
        "You are a helpful curator of information on the Internet. "
                                  
        "Suggest the best website for finding something that would contribute meaningfully "
        "to the conversation. "
                                  
        "Limit your suggestions to websites that work well with text-based browsers. "
    ).strip()

    if candidates:
        prompt += (
                "\nConsider the following candidate(s):\n" + '\n'.join(f"- {c}" for c in candidates)
        )

    try:
        response = await async_openai_client.chat.completions.create(
            messages=[{"role": "system", "content": prompt}] + messages,
            model=germ_settings.CURATION_MODEL,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "website",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "domains": {
                                "type": "array",
                                "description": "A list of website domain names.",
                                "items": {"type": "string"},
                            }
                        },
                        "required": ["domains"]
                    }
                }
            },
            n=1, timeout=10)
        suggestions = json.loads(response.choices[0].message.content)
        logger.info(f"Best info source suggestions: {suggestions['domains']}")
        assert "domains" in suggestions, "Response does not contain 'domains'"
        # TODO: Return a follow-up task based on the suggestions
        return suggestions
    except Exception:
        logger.error(f"Could not get best info source suggestions: {format_exc()}")
        return {"domains": []}


async def suggest_search_query(
        messages: list[dict[str, str]],
        candidates: Iterable[str],
) -> dict[str, list[str]]:
    prompt = (
        "You are a helpful expert on the use of search engines. "

        "Suggest one or more keyword queries that are likely to yield the best search "
        "results related to the conversation. "
    ).strip()

    if candidates:
        prompt += (
            "\nConsider the following candidate(s):\n" + '\n'.join(f"- {c}" for c in candidates)
        )

    suggestions = []
    while not suggestions:
        try:
            response = await async_openai_client.chat.completions.create(
                messages=[{"role": "system", "content": prompt}] + messages,
                model=germ_settings.CURATION_MODEL,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "keywords",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "queries": {
                                    "type": "array",
                                    "description": "A list of keyword queries.",
                                    "items": {"type": "string"}
                                }
                            },
                            "required": ["queries"]
                        }
                    }
                },
                n=1, timeout=10)
            response_content = json.loads(response.choices[0].message.content)
            assert "queries" in response_content, "Response does not contain 'queries'"
            suggestions.extend(response_content['queries'])
        except Exception:
            logger.error(f"Could not get search query suggestions: {format_exc()}")
    return {"queries": suggestions}


async def summarize_message_received(
        messages: list[dict[str, str]]
) -> dict[str, str]:
    prompt = (
        "You are an efficient assistant for message summarization. "

        "Don't respond to the user but, instead, finish the sentence \"The user ...\" by "
        "summarizing the intent of the most recent user message as succinctly as possible. "

        "Choose your phrasing to effectively capture meaning for comparing similarity and recall "
        "using semantic embeddings. "
    ).strip()

    summary: str | None = None
    while not summary:
        try:
            response = await async_openai_client.chat.completions.create(
                messages=[{"role": "system", "content": prompt}] + messages,
                model=germ_settings.SUMMARY_MODEL,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "summary",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "summary": {
                                    "type": "string",
                                    "description": "A summary of the most recent user message.",
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


async def summarize_message_sent(
        messages: list[dict[str, str]]
) -> dict[str, list[str]]:
    prompt = (
        "You are an efficient assistant for message summarization. "

        "Summarize the contents of the assistant message you just sent. "

        "Use statements that effectively capture meaning for comparing similarity and recall "
        "using semantic embeddings. "

        "Start each statement with \"I ...\", then complete the sentence by explaining "
        "what you (the assistant) did or said. "
        
        "When needed, refer to the user as \"the user\". "
    ).strip()

    summary: list[str] = []
    while not summary:
        try:
            response = await async_openai_client.chat.completions.create(
                messages=[{"role": "system", "content": prompt}] + messages,
                model=germ_settings.SUMMARY_MODEL,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "summary",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "statements": {
                                    "type": "array",
                                    "description": "A list of statements summarizing your message to the user.",
                                    "items": {"type": "string"}
                                }
                            },
                            "required": ["statements"]
                        }
                    }
                },
                n=1, timeout=10)
            response_content = json.loads(response.choices[0].message.content)
            assert "statements" in response_content, "Response does not contain 'statements'"
            summary.extend(response_content["statements"])
        except Exception:
            logger.error(f"Could not get response summary: {format_exc()}")
    return {"statements": summary}
