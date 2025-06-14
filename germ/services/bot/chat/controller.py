import aiohttp
import asyncio
import logging
from datetime import datetime
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from germ.api.models import ChatRequest, ChatResponse
from germ.database.neo4j import KnowledgeGraph
from germ.observability.annotations import measure_exec_seconds
from germ.services.bot.websocket import (WebSocketDisconnectEventHandler, WebSocketReceiveEventHandler,
                                         WebSocketSendEventHandler, WebSocketSender, WebSocketSessionMonitor)
from germ.services.models.predict.multi_predict import log_pos_labels
from germ.settings import germ_settings
from germ.utils.parsers import PageElementType, ParsedMarkdownPage, extract_markdown_page_elements

logger = logging.getLogger(__name__)


class MessageMeta(BaseModel):
    doc: ParsedMarkdownPage
    pos: list[dict[str, str | list[str]]]
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
        self.sig_to_conversation_id: dict[str, set] = {}
        self.sig_to_message_meta: dict[str, MessageMeta] = {}

    async def on_disconnect(self, conversation_id: int):
        pass

    @measure_exec_seconds(use_logging=True, use_prometheus=True)
    async def on_receive(self, user_id: int, conversation_id: int, dt_created: datetime, text_sig: str,
                         chat_request: ChatRequest, ws_sender: WebSocketSender):
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = {
                # Indexes user_id and text_sig by dt_created
                "messages": {},
                "token_size": 0,
            }
        self.conversations[conversation_id]["messages"][int(dt_created.timestamp())] = {
            "user_id": user_id, "text_sig": text_sig
        }
        self.sig_to_conversation_id.get(text_sig, set()).add(conversation_id)

        if text_sig not in self.sig_to_message_meta:
            parsed_message = await run_in_threadpool(
                extract_markdown_page_elements, chat_request.messages[-1].content
            )
            text_embs, pos = await asyncio.gather(*[
                get_text_embedding(parsed_message.text),
                get_pos_labels(parsed_message.text)
            ])
            self.sig_to_message_meta[text_sig] = meta = MessageMeta(
                doc=parsed_message, pos=pos, text_embs=text_embs["embeddings"]
            )
        else:
            meta = self.sig_to_message_meta[text_sig]

        for element_idx, element in enumerate(meta.doc.scaffold):
            if element.type == PageElementType.CODE_BLOCK:
                pass  # TODO
            elif element.type == PageElementType.LIST:
                for item_idx, item in enumerate(element.items):
                    for sentence_idx in item.text:
                        logger.info(f"{meta.doc.text[sentence_idx]} >> {meta.text_embs[sentence_idx]}")
                        log_pos_labels(meta.pos[sentence_idx])
            elif element.type == PageElementType.PARAGRAPH:
                for sentence_idx in element.text:
                    logger.info(f"{meta.doc.text[sentence_idx]} >> {meta.text_embs[sentence_idx]}")
                    log_pos_labels(meta.pos[sentence_idx])
                    #await self.knowledge_graph.match_synset(tokens=meta.pos[sentence_idx]["tokens"])

        # Send to LLM
        await self.delegate.on_receive(user_id, conversation_id, dt_created, text_sig, chat_request, ws_sender)

    async def on_send(self, conversation_id: int, dt_created: datetime, text_sig: str,
                      chat_response: ChatResponse, received_message_dt_created: datetime = None):
        pass

    async def on_tick(self, conversation_id: int, ws_sender: WebSocketSender):
        logger.info(f"conversation {conversation_id} is still active")


async def get_pos_labels(texts: list[str]):
    async with aiohttp.ClientSession() as session:
        async with session.post(f"http://{germ_settings.MODEL_SERVICE_ENDPOINT}/text/classification/ud",
                                json={"texts": texts}) as response:
            return await response.json()


async def get_text_embedding(texts: list[str]):
    async with aiohttp.ClientSession() as session:
        async with session.post(f"http://{germ_settings.MODEL_SERVICE_ENDPOINT}/text/embedding",
                                json={"texts": texts}) as response:
            return await response.json()
