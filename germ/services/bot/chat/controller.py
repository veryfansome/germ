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
from germ.utils import find_near_dup_pairs, normalize_embeddings
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
        summary_text = f"The user {message_summary['statement']}"
        logger.info(f"User message summary:\n - {summary_text}")
        logger.info(f"User intent labels: {''.join(f'\n - {l}' for l in user_intent_labels['intents'])}")

        # Generate embeddings on summarized user message
        summary_emb = await run_in_threadpool(
            normalize_embeddings, np.array(await get_text_embedding([summary_text]), dtype=np.float32)
        )
        summary_emb_floats = summary_emb[0].tolist()

        # TODO: handle if situations that don't require information search

        # Recall
        min_similarity = 0.75
        (
            recalled_bot_message_summaries,
            recalled_user_message_summaries,
        ) = await asyncio.gather(
            self.knowledge_graph.match_bot_message_summaries_by_similarity_to_query_vector(
                summary_emb_floats, k=15, min_similarity=min_similarity,
            ),
            self.knowledge_graph.match_user_message_summaries_by_similarity_to_query_vector(
                user_id, summary_emb_floats, k=15, min_similarity=min_similarity,
            ),
        )
        recalled_keyword_phrases = await self.knowledge_graph.match_keyword_phrases_by_similarity_to_query_vector(
            summary_emb_floats, [
                {
                    "conversation_id": struct["conversation_id"],
                    "dt_created": struct["dt_created"],
                    "score": struct["score"],
                } for struct in recalled_user_message_summaries
            ], alpha=0.7, k=5,
        )
        logger.info(f"Recalled bot message summaries: {''.join(
            ('\n - ' + str((r['score'], r['text'], r['conversation_id'], r['dt_created'])))
            for r in recalled_bot_message_summaries if r["conversation_id"] != conversation_id
        )}")
        logger.info(f"Recalled user message summaries: {''.join(
            ("\n - " + str((r['score'], r['text'], r['conversation_id'], r['dt_created'])))
            for r in recalled_user_message_summaries if r["conversation_id"] != conversation_id
        )}")
        logger.info(f"Recalled keyword phrases: {''.join(
            ("\n - " + str((r['score'], r['text']))) for r in recalled_keyword_phrases
        )}")

        # TODO: Pull from Neo4j
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
            keyword_phrase_suggestions
        ) = await asyncio.gather(
            suggest_best_online_info_source(filtered_messages, info_source_candidates),
            suggest_keyword_phrases(filtered_messages, [r["text"] for r in recalled_keyword_phrases])
        )
        logger.info(f"Info source suggestions: {info_source_suggestions['domains']}")
        logger.info(f"Search keyword suggestions: {keyword_phrase_suggestions['phrases']}")

        doc = await doc_parse_task
        message_meta = ChatMessageMetadata(
            doc=doc,
            info_source_suggestions=info_source_suggestions["domains"],
            keyword_phrase_suggestions=keyword_phrase_suggestions["phrases"],
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
            self.knowledge_graph.add_user_message_intent(conversation_id, dt_created, user_intent_labels['intents']),
            self.persist_message_summary(
                conversation_id, dt_created, 0,
                summary_text, summary_emb_floats, recalled_user_message_summaries
            ),
            self.persist_keyword_phrase_embeddings(
                conversation_id, dt_created, keyword_phrase_suggestions["phrases"]
            )
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

        # Recall similar message summaries for deduplication
        min_similarity = 0.75
        recall_tasks = {}
        for idx, emb in enumerate(text_embs):
            recall_tasks[idx] = asyncio.create_task(
                self.knowledge_graph.match_bot_message_summaries_by_similarity_to_query_vector(
                    emb.tolist(), k=3, min_similarity=min_similarity,
                )
            )

        # Call persist_message_summary() with recalled summaries
        persist_summary_tasks = []
        for idx, task in recall_tasks.items():
            recalled_structs = await task
            logger.info(f"Recalled {len(recalled_structs)} summaries for deduplication")
            persist_summary_tasks.append(asyncio.create_task(
                self.persist_message_summary(
                    conversation_id, dt_created, idx,
                    summary_statements[idx], text_embs[idx].tolist(), recalled_structs,
                    sim_threshold=min_similarity,
                )
            ))

        doc = await doc_parse_task
        message_meta = ChatMessageMetadata(
            doc=doc,
            user_id=0
        )

        await asyncio.gather(
            self.conversations[conversation_id].add_message(dt_created, message_meta, log=True),
            *persist_summary_tasks,
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

    async def persist_message_summary(
            self, conversation_id: int, dt_created: datetime, position: int,
            summary_text: str, summary_emb_floats: list[float], recalled_summary_structs,
            sim_threshold: float = 0.7,
    ):
        create_new_summary = True
        if recalled_summary_structs:
            candidates = [(summary_text, summary_emb_floats)]
            seen = set()
            for struct in recalled_summary_structs:
                if summary_text == struct["text"]:
                    # Exact dup, link message to existing summary and return
                    await self.knowledge_graph.link_chat_message_to_summary(
                        conversation_id, dt_created, position, summary_text
                    )
                    return
                elif struct["text"] not in seen:
                    # If not exact dup, append for additional processing
                    candidates.append((struct["text"], struct["embedding"]))
                    seen.add(struct["text"])

            # Identify near duplicate pairs using embeddings
            near_dup_pairs = await run_in_threadpool(
                find_near_dup_pairs, [t[1] for t in candidates],
                normalize=False,  # Since already normalized
                sim_threshold=sim_threshold,
            )

            # Get judgment call on near duplicates
            pending_judgments = {}
            for pair_idx, pair in enumerate(near_dup_pairs):  # [(x_idx, y_idx, score), ...]
                if pair[0] != 0 and pair[1] != 0:
                    # We care only about pairs with the received summary at combined_embs position 0
                    continue
                x = summary_text if pair[0] == 0 else candidates[pair[0]][0]
                y = summary_text if pair[1] == 0 else candidates[pair[1]][0]
                pending_judgments[pair_idx] = (asyncio.create_task(dedup_summaries([x, y])), x, y)
            for pair_idx, (judgment_task, x, y) in pending_judgments.items():
                judgment = await judgment_task
                logger.info(f"Merge near-duplicate summaries: {judgment['judgment']}, {(x, y)}")
                if judgment["judgment"] == "yes":
                    # TODO:
                    #   - combine or choose the new summary text
                    #   - pool embeddings?
                    await self.knowledge_graph.link_chat_message_to_summary(
                        conversation_id, dt_created, position, x if x != summary_text else y
                    )
                    create_new_summary = False

        if create_new_summary:
            await self.knowledge_graph.add_summary(conversation_id, dt_created, {
                summary_text: (position, summary_emb_floats)
            }),

    async def persist_keyword_phrase_embeddings(
            self, conversation_id, dt_created: datetime, new_suggestions: list[str]
    ):
        new_suggestions_embedding = await run_in_threadpool(
            normalize_embeddings, np.array(await get_text_embedding(new_suggestions), dtype=np.float32)
        )
        await self.knowledge_graph.add_keyword_phrases(
            conversation_id, dt_created, {
                txt: new_suggestions_embedding[idx].tolist() for idx, txt in enumerate(new_suggestions)
            }
        )


class ChatMessageMetadata(BaseModel):
    doc: ParsedDoc
    info_source_suggestions: list[str] | None = None
    message_summary: str | None = None
    keyword_phrase_suggestions: list[str] | None = None
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


async def dedup_summaries(statements: list[str]):
    prompt = (
        "You are a deduper of similar statements summarizing chat messages. "

        "Determine if the given statements describe the same situation or user intent. "
        
        "If the statements can be deduplicated, answer 'yes', otherwise answer 'no'. "
        
        "For example, when comparing \"The user requested a joke\" and \"The user requested another joke\", "
        "the answer is 'yes' because in both cases, the user wanting a joke is the underlying situation. "
    ).strip()

    judgment: str | None = None
    while not judgment:
        try:
            response = await async_openai_client.chat.completions.create(
                messages=[{"role": "system", "content": prompt}, {
                    "role": "user", "content": (
                        "Summary statements:"
                        "".join([f"\n{num}. {s}" for num, s in enumerate(statements, 1)])
                    )
                }],
                model=germ_settings.CURATION_MODEL,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "equality",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "judgment": {
                                    "type": "string",
                                    "description": "A list of website domain names.",
                                    "enum": ["yes", "no"]
                                }
                            },
                            "required": ["judgment"]
                        }
                    }
                },
                n=1, timeout=10)
            response_content = json.loads(response.choices[0].message.content)
            assert "judgment" in response_content, "Response does not contain 'judgment'"
            normalized_content = response_content["judgment"].lower().strip()
            assert normalized_content in {"yes", "no"}, f"Expected 'yes' or 'no', got '{normalized_content}'"
            judgment = normalized_content
        except Exception:
            logger.error(f"Could not get deduplication judgment: {format_exc()}")
    return {"judgment": judgment}


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

    seen = set()
    suggestions = []
    while not suggestions:
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
            response_content = json.loads(response.choices[0].message.content)
            assert "domains" in response_content, "Response does not contain 'domains'"
            for dom in response_content["domains"]:
                if dom.strip() == "":
                    continue
                elif dom in seen:
                    continue
                else:
                    seen.add(dom)
                    suggestions.append(dom)
        except Exception:
            logger.error(f"Could not get best info source suggestions: {format_exc()}")
    return {"domains": suggestions}


async def suggest_keyword_phrases(
        messages: list[dict[str, str]],
        candidates: Iterable[str],
) -> dict[str, list[str]]:
    prompt = (
        "You are a helpful expert on the use of search engines. "

        "Suggest one or more keyword phrases that are likely to yield the best search "
        "results related to the conversation. "
    ).strip()

    if candidates:
        prompt += (
            "\nConsider the following candidate(s):\n" + '\n'.join(f"- {c}" for c in candidates)
        )

    seen = set()
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
                                "phrases": {
                                    "type": "array",
                                    "description": "A list of keyword phrases.",
                                    "items": {"type": "string"}
                                }
                            },
                            "required": ["phrases"]
                        }
                    }
                },
                n=1, timeout=10)
            response_content = json.loads(response.choices[0].message.content)
            assert "phrases" in response_content, "Response does not contain 'phrases'"
            for qry in response_content["phrases"]:
                if qry.strip() == "":
                    continue
                elif qry in seen:
                    continue
                else:
                    seen.add(qry)
                    suggestions.append(qry)
        except Exception:
            logger.error(f"Could not get keyword phrase suggestions: {format_exc()}")
    return {"phrases": suggestions}


async def summarize_message_received(
        messages: list[dict[str, str]]
) -> dict[str, str]:
    prompt = (
        "You are an efficient assistant for message summarization. "

        "Don't respond to the user but, instead, finish the sentence \"The user ...\" by "
        "summarizing the substance of the most recent user message. "

        "Focus only on what was said, i.e. the core inquiries, imperatives, or ideas the user conveyed. "

        "Do not invent or generalize beyond what was said. "
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
    return {"statement": summary}


async def summarize_message_sent(
        messages: list[dict[str, str]]
) -> dict[str, list[str]]:
    prompt = (
        "You are an efficient assistant for message summarization. "

        "Summarize the substance of the most recent message the assistant (you) sent to the user. "

        "Focus only on what was said, i.e. the core ideas the assistant conveyed. "

        "Do not invent or generalize beyond what was said. "

        "Multiple statement may be appropriate, but only if what was said cannot be summarized "
        "simply using a single statement. "

        "Start each statement with \"I ...\", then complete the sentence by explaining "
        "what you (the assistant) did or said. "
    ).strip()

    seen = set()
    statements: list[str] = []
    while not statements:
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
                                    "description": ("A list of simple statements summarizing core ideas "
                                                    "conveyed to the user."),
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
            for stmt in response_content["statements"]:
                if stmt.strip() == "":
                    continue
                elif stmt in seen:
                    continue
                else:
                    seen.add(stmt)
                    statements.append(stmt)
        except Exception:
            logger.error(f"Could not get response summary: {format_exc()}")
    return {"statements": statements}
