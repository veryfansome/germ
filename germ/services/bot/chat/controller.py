import aiofiles
import aiofiles.os
import asyncio
import json
import logging
import numpy as np
import re
import time
from datetime import datetime, timezone
from opentelemetry import trace
from pydantic import BaseModel
from re import Pattern
from starlette.concurrency import run_in_threadpool
from traceback import format_exc
from typing import Iterable

from germ.api.models import ChatRequest, ChatResponse
from germ.browser import FetchResult, WebBrowser
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

tracer = trace.get_tracer(__name__)


class ChatController(WebSocketDisconnectEventHandler, WebSocketReceiveEventHandler,
                     WebSocketSendEventHandler, WebSocketSessionMonitor):
    def __init__(self, knowledge_graph: KnowledgeGraph, web_browser: WebBrowser):
        self.conversations: dict[int, Conversation] = {}
        self.knowledge_graph = knowledge_graph
        self.web_browser = web_browser

    async def fetch_online_info(
            self, filtered_messages: list[dict[str, str]], user_agent: str, extra_headers: dict[str, str]
    ):
        info_source_suggestions = await suggest_best_online_info_source(filtered_messages, [])
        logger.info(f"Info source suggestions: {info_source_suggestions['urls']}")

        coroutines = [self.web_browser.fetch(url, user_agent, extra_headers)
                      for url in info_source_suggestions["urls"]]
        results = await asyncio.gather(*coroutines, return_exceptions=True)
        for url, result in zip(info_source_suggestions["urls"], results):
            if isinstance(result, Exception):
                logger.error(f"Failed to fetch {url}", exc_info=result)
                continue
            if logger.level == logging.DEBUG:
                logger.debug(f"Fetched {len(result.text)} characters of page text: "
                             f"{result.status_code} {result.content_type} {result.extraction_status} {result.url}")

    async def load_conversations(self, user_id: int, conversation_ids: Iterable[int]):
        if not conversation_ids:
            return

        logger.info(f'Loading conversations: {conversation_ids}')
        chat_message_structs = await self.knowledge_graph.match_chat_messages_by_conversation_id(conversation_ids)
        for (conversation_id, year, month, day) in {
            (s["conversation_id"], s["dt_created"].year, s["dt_created"].month, s["dt_created"].day)
            for s in chat_message_structs
        }:
            if conversation_id not in self.conversations:
                self.conversations[conversation_id] = Conversation(conversation_id=conversation_id, user_id=user_id)

            storage_path = get_storage_path(user_id, year, month, day)
            data_file_name = get_data_file_name(conversation_id, year, month, day)
            async with aiofiles.open(f"{storage_path}/{data_file_name}", encoding="utf-8") as f:
                async for line in f:
                    data = json.loads(line)
                    logger.debug(f'Loading data: {data}')
                    self.conversations[conversation_id].messages[datetime.fromtimestamp(data[0], tz=timezone.utc)] = (
                        data[1], ParsedDoc.model_validate_json(data[2])
                    )

    async def on_disconnect(self, conversation_id: int):
        pass

    @measure_exec_seconds(use_logging=True, use_prometheus=True)
    async def on_receive(self, session, conversation_id: int, dt_created: datetime,
                         chat_request: ChatRequest, ws_sender: WebSocketSender):
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = Conversation(
                conversation_id=conversation_id, user_id=session["user_id"],
            )

        if chat_request.uploaded_filenames:
            for filename in chat_request.uploaded_filenames:
                # TODO: handle uploaded files
                pass

        message_doc = await run_in_threadpool(ParsedDoc.from_text, chat_request.messages[-1].content)
        self.conversations[conversation_id].messages[dt_created] = (session["user_id"], message_doc)
        # TODO: code block summarization

        filtered_messages: list[dict[str, str]] = [m.model_dump() for m in chat_request.messages if m.role != "system"]

        # Label user intent in background
        user_intent_labels_task = asyncio.create_task(UserIntentClassifier.classify_user_intent(filtered_messages))

        # Summarized user message
        message_summary = await summarize_message_received(filtered_messages)
        summary_text = f"The user {message_summary['statement']}"
        logger.info(f"User message summary:\n - {summary_text}")

        # Generate embeddings on summarized user message
        summary_emb = await run_in_threadpool(
            normalize_embeddings, np.array(await get_text_embedding([summary_text]), dtype=np.float32)
        )
        summary_emb_floats = summary_emb[0].tolist()

        ##
        # Recall Phase 1

        # Get similar messages from same user
        min_similarity = 0.75
        (
            recalled_user_message_structs,  # user messages similar to summary_emb
        ) = await asyncio.gather(
            self.knowledge_graph.match_user_message_summaries_by_similarity_to_query_vector(
                session["user_id"], summary_emb_floats, k=15, min_similarity=min_similarity,
            ),
        )
        if logger.level == logging.DEBUG:
            logger.debug(f"Recalled user message summaries: {''.join(
                ("\n - " + str((s['score'], s['text'], s['conversation_id'], s['dt_created'])))
                for s in recalled_user_message_structs if s["conversation_id"] != conversation_id
            )}")

        # Pull old conversations from storage
        pending_tasks = [
            asyncio.create_task(self.load_conversations(
                session["user_id"],
                {
                    s["conversation_id"] for s in recalled_user_message_structs
                    if s["conversation_id"] != conversation_id and s["conversation_id"] not in self.conversations
                }
            ))
        ]

        ##
        # Recall Phase 2

        # Get replies linked to similar user messages
        linked_replies_task = asyncio.create_task(
            fetch_relevant_linked_replies(filtered_messages, asyncio.create_task(
                self.knowledge_graph.match_reply_summaries_by_similarity_to_query_vector(
                    summary_emb_floats, [
                        # Exclude messages received from current conversation
                        s for s in recalled_user_message_structs if s["conversation_id"] != conversation_id
                    ], alpha=0.5, k=30,
                )
            ))
        )
        pending_tasks.append(linked_replies_task)

        # Get intent labels
        user_intent_labels = await user_intent_labels_task
        logger.info(f"User intent labels: {''.join(f'\n - {l}' for l in user_intent_labels['intents'])}")

        has_informational_intent = False
        has_instrumental_intent = False
        for intent_category, intent in user_intent_labels["intents"]:
            if intent_category == "informational":
                has_informational_intent = True
            elif intent_category == "instrumental":
                has_instrumental_intent = True

        # Intent-specific tasks
        online_info_task = None
        keyword_phrase_suggestions_task = None

        if has_informational_intent or has_instrumental_intent:
            online_info_task = asyncio.create_task(self.fetch_online_info(
                filtered_messages, session["user_agent"], session["headers"]
            ))
            pending_tasks.append(online_info_task)

            keyword_phrase_suggestions_task = asyncio.create_task(
                fetch_keyword_phrase_suggestions(filtered_messages, asyncio.create_task(
                    self.knowledge_graph.match_keyword_phrases_by_similarity_to_query_vector(
                        # Ok to include previous phrases from current conversation
                        summary_emb_floats, recalled_user_message_structs, alpha=0.5, k=5,
                    )
                ))
            )
            pending_tasks.append(keyword_phrase_suggestions_task)

        await asyncio.gather(*pending_tasks)

        await asyncio.gather(
            # Complete chat request
            (
                openai_handlers.ReasoningChatModelEventHandler() if germ_settings.OPENAI_REASONING_MODEL
                else openai_handlers.ChatModelEventHandler()
            ).on_receive(session, conversation_id, dt_created, chat_request, ws_sender),

            # House-keeping
            self.knowledge_graph.add_user_message_intent(conversation_id, dt_created, user_intent_labels['intents']),
            self.persist_keyword_phrase_embeddings(
                conversation_id, dt_created,
                [] if not keyword_phrase_suggestions_task else keyword_phrase_suggestions_task.result()["phrases"]
            ),
            self.persist_message(session["user_id"], conversation_id, dt_created, message_doc),
            self.persist_message_summary(
                conversation_id, dt_created, 0,
                summary_text, summary_emb_floats, recalled_user_message_structs
            ),
        )

    async def on_send(self, conversation_id: int, dt_created: datetime,
                      chat_response: ChatResponse, received_message_dt_created: datetime = None):
        if conversation_id not in self.conversations:
            logger.error(f"Conversation {conversation_id} not found.")
            return

        message_doc = await run_in_threadpool(ParsedDoc.from_text, chat_response.content)
        self.conversations[conversation_id].messages[dt_created] = (0, message_doc)

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

        await asyncio.gather(
            self.persist_message(0, conversation_id, dt_created, message_doc),
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
        # TODO: if a conversation has no user messages and has been idle for some time, recall what is known about
        #       the user and attempt to start a conversation.

    async def persist_message(self, user_id: int, conversation_id: int, dt_created: datetime, message_doc: ParsedDoc):
        storage_path = get_storage_path(self.conversations[conversation_id].user_id,
                                        dt_created.year, dt_created.month, dt_created.day)
        await aiofiles.os.makedirs(storage_path, exist_ok=True)

        data_file_name = get_data_file_name(conversation_id, dt_created.year, dt_created.month, dt_created.day)
        async with aiofiles.open(f"{storage_path}/{data_file_name}", mode="a", encoding="utf-8") as f:
            await f.write(f"{json.dumps([
                dt_created.timestamp(),
                user_id,
                message_doc.model_dump_json(exclude_none=True)
            ])}\n")
            await f.flush()

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
        if not new_suggestions:
            return

        new_suggestions_embedding = await run_in_threadpool(
            normalize_embeddings, np.array(await get_text_embedding(new_suggestions), dtype=np.float32)
        )
        await self.knowledge_graph.add_keyword_phrases(
            conversation_id, dt_created, {
                txt: new_suggestions_embedding[idx].tolist() for idx, txt in enumerate(new_suggestions)
            }
        )


class Conversation(BaseModel):
    conversation_id: int
    messages: dict[datetime, tuple[int, ParsedDoc]] = {}
    user_id: int

    def hydrate_messages(self) -> list[dict[str, str]]:
        return [
            {
                "role": ("assistant" if user_id == 0 else "user"),
                "content": doc.restore()
            }
            for _, (user_id, doc) in self.messages.items()
        ]


async def dedup_summaries(statements: list[str]):
    prompt = (
        "You are a deduper for summary statements of chat messages. "

        "\n\nTask: "
        "\n- Decide if the provided statements describe the same situation or intent. "
        "\n- If all statements can be deduplicated into one canonical underlying situation or intent, answer 'yes'. "
        "\n- If any statement materially differs (topic, target, action, constraints, etc.), answer 'no'. "
        "\n- If uncertain, answer 'no' to prioritize precision over recall. "

        "\n\nGuidelines: "
        "\n- Statements involving the same actors, actions, and primary topics/goals should be treated as the same. "
        "\n- Statements with small wording differences (tense, politeness, count/plurality, etc.) that don't change the core intent should be treated as the same. "
        "\n- 'another' or 'again' still maps to the same underlying situation or intent if nothing else differs. "

        "\n\nOutput: "
        "\n- Return only a JSON object conforming to the provided schema with a 'judgment' attribute that is either 'yes' or 'no'. "

        "\n\nExamples that are the same: "
        "\n- \"The user requested a joke.\" vs \"The user wants another joke.\" => yes "
        "\n- \"Summarize the article.\" vs \"Provide a brief summary of the article.\" => yes "

        "\n\nExamples that are not the same: "
        "\n- \"Translate this to Spanish.\" vs \"Translate this to French.\" => no "
        "\n- \"Summarize the article.\" vs \"Explain the article.\" => no "
        "\n- \"Help me place an order.\" vs \"Help me cancel an order.\" => no "
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
                model=germ_settings.OPENAI_DEDUP_MODEL,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "equality",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "judgment": {
                                    "type": "string",
                                    "description": "Judgment of whether to deduplicate the statements.",
                                    "enum": ["yes", "no"]
                                }
                            },
                            "additionalProperties": False,
                            "required": ["judgment"]
                        }
                    }
                },
                n=1,
                seed=germ_settings.OPENAI_SEED,
                timeout=10.0,
            )
            response_content = json.loads(response.choices[0].message.content)
            assert "judgment" in response_content, "Response does not contain 'judgment'"
            normalized_content = response_content["judgment"].lower().strip()
            assert normalized_content in {"yes", "no"}, f"Expected 'yes' or 'no', got '{normalized_content}'"
            judgment = normalized_content
        except Exception:
            logger.error(f"Could not get deduplication judgment: {format_exc()}")
    return {"judgment": judgment}


async def fetch_keyword_phrase_suggestions(
        filtered_messages: list[dict[str, str]],
        keyword_phrase_recall_task
):
    recalled_keyword_phase_structs = await keyword_phrase_recall_task
    if logger.level == logging.DEBUG:
        logger.debug(f"Recalled keyword phrases: {''.join(
            ("\n - " + str((k['score'], k['text']))) for k in recalled_keyword_phase_structs
        )}")

    keyword_phrase_suggestions = await suggest_keyword_phrases(filtered_messages, [k["text"] for k in recalled_keyword_phase_structs])
    logger.info(f"Search keyword suggestions: {keyword_phrase_suggestions['phrases']}")
    return keyword_phrase_suggestions


async def fetch_relevant_linked_replies(
        filtered_messages: list[dict[str, str]],
        linked_reply_recall_task
):
    recalled_linked_reply_structs = await linked_reply_recall_task
    if logger.level == logging.DEBUG:
        logger.debug(f"Recalled linked reply summaries: {''.join(
            ('\n - ' + str((s['score'], s['text'], s['conversation_id'], s['dt_created'])))
            for s in recalled_linked_reply_structs
        )}")

    bot_summary_texts = []
    for s in recalled_linked_reply_structs:
        if s["text"] not in bot_summary_texts:
            bot_summary_texts.append(s["text"])

    filtered_summaries = await filter_relevant_summaries(filtered_messages, bot_summary_texts)
    logger.info(f"Relevant summaries: {''.join(
        f'\n - {bot_summary_texts[idx]}' for idx in filtered_summaries['items']
    )}")
    # TODO: review messages linked to recalled summaries


async def filter_relevant_summaries(
        messages: list[dict[str, str]],
        recalled_summaries: list[str],
) -> dict[str, list[int]]:
    if not recalled_summaries:
        return {"items": []}

    prompt = (
        "You are a relevance gate for recalled chat message summaries from past conversations. "

        "\n\nTask: "
        "\n- Consider the user's current intention or objective based on their most recent message. "
        "\n- Decide which past messages should be retrieved and reviewed based on their summary's relevance to the current conversation. "
        
        "\n\nGuidelines: "
        "\n- Messages should be reviewed: "
        "\n  - If reviewing them should result in a more consistent response. "
        "\n  - If they should contain relevant facts, contexts, or directly reusable content. "
        "\n- Messages should not be reviewed: "
        "\n  - If reviewing them is unlikely to result in a higher quality or more informed response. "
        "\n  - If they are about unrelated subject matter or only some-what related subject matter. "

        "\n\nOutput: "
        "\n- Return only a JSON object conforming to the provided schema with an 'items' attribute that lists item numbers from the following recalled summaries. "
        
        "\n\nRecalled summaries: "
    ).strip()
    prompt += ("".join([f"\n{num}. {s}" for num, s in enumerate(recalled_summaries, 1)]))  # Shift by 1

    relevant_items: list[int] | None = None
    while relevant_items is None:
        try:
            response = await async_openai_client.chat.completions.create(
                messages=[{"role": "system", "content": prompt}] + messages,
                model=germ_settings.OPENAI_CURATION_MODEL,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "relevant_summaries",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "items": {
                                    "type": "array",
                                    "description": "Item numbers of the selected summaries.",
                                    "items": {"type": "number"},
                                    "minItems": 0,
                                    "uniqueItems": True,
                                }
                            },
                            "additionalProperties": False,
                            "required": ["judgment"]
                        },
                    }
                },
                n=1,
                seed=germ_settings.OPENAI_SEED,
                timeout=10.0,
            )
            response_content = json.loads(response.choices[0].message.content)
            assert "items" in response_content, "Response does not contain 'items'"
            relevant_items = [
                (n - 1)  # Unshift by 1
                for n in response_content["items"]
            ]
        except Exception:
            logger.error(f"Could not get relevant items: {format_exc()}")
    return {"items": relevant_items}


def get_data_file_name(conversation_id: int, year: int, month: int, day: int):
    return f"{conversation_id:011d}_{year}{month:02d}{day}.ndjson"


def get_storage_path(user_id: int, year: int, month: int, day: int,
                     data_dir: str = f"{germ_settings.GERM_DATA_DIR}/chat"):
    return f"{data_dir}/{user_id:05d}/{year}/{month:02d}/{day}"


async def suggest_best_online_info_source(
        messages: list[dict[str, str]],
        candidates: list[str]
) -> dict[str, list[str]]:
    prompt = (
        "You are a curator of authoritative information on the Internet. "

        "\n\nTask: "
        "\n- Choose the best webpages for finding authoritative information to assist the user. "
        
        "\n\nGuidelines: "
        "\n- Prioritize well maintained, trusted websites that are more likely to have stable URLs, and accurate, up-to-date information. "
        "\n- Prioritize pages that don't require logins and are not gated behind paywalls. "
        "\n- In cases where multiple URLs lead to the same content, pick only one. "

        "\n\nOutput: "
        "\n- Return only a JSON object conforming to the provided schema with a 'urls' attribute that is a list of recommended webpage URLs. "
    ).strip()

    #if candidates:
    #    prompt += (
    #            "\nConsider the following candidate(s):\n" + '\n'.join(f"- {c}" for c in candidates)
    #    )

    seen = set()
    suggestions = []
    while not suggestions:
        try:
            response = await async_openai_client.chat.completions.create(
                messages=[{"role": "system", "content": prompt}] + messages,
                model=germ_settings.OPENAI_CURATION_MODEL,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "webpage",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "urls": {
                                    "type": "array",
                                    "description": "A list of recommended webpage URLs.",
                                    "items": {"type": "string"},
                                }
                            },
                            "additionalProperties": False,
                            "required": ["urls"]
                        },
                        "strict": True,
                    }
                },
                n=1,
                seed=germ_settings.OPENAI_SEED,
                timeout=30.0,
            )
            response_content = json.loads(response.choices[0].message.content)
            assert "urls" in response_content, "Response does not contain 'urls'"
            for dom in response_content["urls"]:
                if dom.strip() == "":
                    continue
                elif dom in seen:
                    continue
                else:
                    seen.add(dom)
                    suggestions.append(dom)
        except Exception:
            logger.error(f"Could not get best info source suggestions: {format_exc()}")
    return {"urls": suggestions}


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
                model=germ_settings.OPENAI_CURATION_MODEL,
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
                            "additionalProperties": False,
                            "required": ["phrases"]
                        },
                        "strict": True,
                    }
                },
                n=1,
                seed=germ_settings.OPENAI_SEED,
                timeout=10.0,
            )
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
        messages: list[dict[str, str]],
        summary_prefix_pattern: Pattern[str] = re.compile(r"^(The user|User):?\s*(?:\.\.\.\s*)?")
) -> dict[str, str]:
    prompt = (
        "You are a summarizer of user chat messages. "

        "\n\nTask: "
        "\n- Summarize the substance of the most recent message sent by the user. "
        "\n- Start your summary with \"The user ...\", then complete the sentence by explaining what the user wants or said. "

        "\n\nGuidelines: "
        "\n- Only summarize the user's message, don't respond to the user. "
        "\n- Voice your summary as the assistant, addressing a person that is not the user. "
        "\n- Focus only on what was said, i.e. the core ideas, imperatives, inquiries, intents, or situations the user conveyed. "
        "\n- Use complete and grammatically correct sentences. "

        "\n\nOutput: "
        "\n- Return only a JSON object conforming to the provided schema with a 'summary' attribute that is your summary statement. "
    ).strip()

    summary: str | None = None
    while not summary:
        try:
            response = await async_openai_client.chat.completions.create(
                messages=[{"role": "system", "content": prompt}] + messages,
                model=germ_settings.OPENAI_SUMMARY_MODEL,
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
                            "additionalProperties": False,
                            "required": ["summary"]
                        },
                        "strict": True,
                    }
                },
                n=1,
                seed=germ_settings.OPENAI_SEED,
                timeout=10.0,
            )
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
        "You are a summarizer of assistant chat messages. "

        "\n\nTask: "
        "\n- Summarize the substance of the most recent message the assistant sent to the user. "
        "\n- Start each statement with \"I ...\", then complete the sentence by explaining what the assistant conveyed. "

        "\n\nGuidelines: "
        "\n- Voice your summary as the assistant, addressing a person that is not the user. "
        "\n- Focus only on what was said, i.e. the core ideas, intent, or situations conveyed. "
        "\n- Multiple statements may be appropriate, but only if what was said cannot be summarized simply using a single statement. "
        "\n- Use complete and grammatically correct sentences. "

        "\n\nOutput: "
        "\n- Return only a JSON object conforming to the provided schema with a 'statements' attribute that is a list containing summary statements. "
    ).strip()

    seen = set()
    statements: list[str] = []
    while not statements:
        try:
            response = await async_openai_client.chat.completions.create(
                messages=[{"role": "system", "content": prompt}] + messages,
                model=germ_settings.OPENAI_SUMMARY_MODEL,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "summary",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "statements": {
                                    "type": "array",
                                    "description": "Statements summarizing what the most recent assistant message conveyed to the user.",
                                    "items": {"type": "string"}
                                }
                            },
                            "additionalProperties": False,
                            "required": ["statements"]
                        },
                        "strict": True,
                    }
                },
                n=1,
                seed=germ_settings.OPENAI_SEED,
                timeout=10.0,
            )
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
