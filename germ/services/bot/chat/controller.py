from datetime import datetime
from starlette.concurrency import run_in_threadpool
import aiohttp
import logging

from germ.api.models import ChatRequest, ChatResponse
from germ.database.neo4j import KnowledgeGraph
from germ.observability.annotations import measure_exec_seconds
from germ.services.bot.websocket import (WebSocketDisconnectEventHandler, WebSocketReceiveEventHandler,
                                         WebSocketSendEventHandler, WebSocketSender, WebSocketSessionMonitor)
from germ.services.models.predict.multi_predict import log_pos_labels
from germ.settings import germ_settings
from germ.utils.parsers import extract_markdown_page_elements

logger = logging.getLogger(__name__)


class MessageMeta:
    def __init__(self, content: str, token_size: int):
        self.content = content
        self.token_size = token_size


class VectorMeta:
    def __init__(self, content: str, token_size: int, vector_id: int, vector, emotions, keywords):
        self.content = content
        self.emotions = emotions
        self.keywords = keywords
        self.token_size = token_size
        self.vector = vector
        self.vector_id = vector_id


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
        self.sig_to_vector: dict[str, VectorMeta] = {}
        self.vector_id_to_sig: dict[int, str] = {}

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
        dt_created_ts = int(dt_created.timestamp())
        self.conversations[conversation_id]["messages"][dt_created_ts] = {
            "user_id": user_id, "text_sig": text_sig
        }

        newest_message_content = chat_request.messages[-1].content
        if text_sig not in self.sig_to_message_meta:
            message_elements = await run_in_threadpool(extract_markdown_page_elements, newest_message_content)
            logger.info(f"parsed message: {message_elements}")

            code_to_enrich = []
            text_to_enrich = []
            scaffolds = []
            for element_idx, element in enumerate(message_elements):
                element_copy = element.copy()
                if element["tag"] == "paragraph":
                    for sentence_idx, sentence in enumerate(element["text"]):
                        text_to_enrich.append(sentence)
                        element_copy["text"][sentence_idx] = len(text_to_enrich) - 1  # Index position
                scaffolds.append(element_copy)
            logger.info(f"text_to_enrich: {text_to_enrich}")
            logger.info(f"scaffolds: {scaffolds}")

            #p_block_sentences = []
            ## TODO: It make sense to do embeddings and POS labels for all sentences from all p_blocks at once
            #for p_block_id, p_block_text in enumerate(p_blocks):
            #    extracted_sentences = await process_p_block(p_block_text)

            #    sentence_embeddings = (await get_text_embedding(extracted_sentences))["embeddings"]
            #    pos_labels = await get_pos_labels(extracted_sentences)
            #    logger.info(f"{len(extracted_sentences)} sentences extracted from conversation {conversation_id}, "
            #                f"message {dt_created_ts}, paragraph {p_block_id}")
            #    logger.info(f"extracted sentences {extracted_sentences}")
            #    for sentence_idx, sentence in enumerate(extracted_sentences):
            #        logger.info(f"{sentence} >> {sentence_embeddings[sentence_idx]}")
            #        log_pos_labels(pos_labels[sentence_idx])
            #    p_block_sentences.append(extracted_sentences)

        else:
            meta = self.sig_to_message_meta[text_sig]
            self.sig_to_conversation_id[text_sig].add(conversation_id)

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
