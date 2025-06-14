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
from germ.utils.parsers import DocElementType, ParsedDoc, parse_markdown_doc

logger = logging.getLogger(__name__)


class MessageMeta(BaseModel):
    doc: ParsedDoc
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
            # Parse message into a ParsedMarkdownPage, which includes a document scaffold (with idx pointers to text),
            # list of text blobs, and list of code blobs.
            parsed_message = await run_in_threadpool(
                parse_markdown_doc, chat_request.messages[-1].content
            )
            # Get embeddings and POS labels
            text_embs, pos = await asyncio.gather(*[
                get_text_embedding(parsed_message.text),
                get_pos_labels(parsed_message.text)
            ])
            # Cache results
            self.sig_to_message_meta[text_sig] = meta = MessageMeta(
                doc=parsed_message, pos=pos, text_embs=text_embs["embeddings"]
            )
        else:
            meta = self.sig_to_message_meta[text_sig]

        # Walk through each element of
        all_hydrated_elements = []
        for element_idx, scaffold_element in enumerate(meta.doc.scaffold):
            hydrated_headings = {}
            for level, heading in scaffold_element.headings.items():
                hydrated_headings[level] = [(meta.text_embs[idx], meta.pos[idx]) for idx in heading.text]

            hydrated_element = []
            if scaffold_element.type == DocElementType.CODE_BLOCK:
                pass  # TODO
            elif scaffold_element.type == DocElementType.LIST:
                for item_idx, item in enumerate(scaffold_element.items):
                    for text_idx in item.text:
                        hydrated_element.append((meta.text_embs[text_idx], meta.pos[text_idx]))
            elif scaffold_element.type == DocElementType.PARAGRAPH:
                for text_idx in scaffold_element.text:
                    hydrated_element.append((meta.text_embs[text_idx], meta.pos[text_idx]))

            for emb, pos in hydrated_element:
                logger.info(f"{pos['text']} >> {emb}")
                log_pos_labels(pos)
                #await self.knowledge_graph.match_synset(tokens=meta.pos[sentence_idx]["tokens"])

            all_hydrated_elements.append((hydrated_headings, hydrated_element))

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
