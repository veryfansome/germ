from datetime import datetime
from starlette.concurrency import run_in_threadpool
import aiohttp
import asyncio
import faiss
import logging
import numpy as np
import tiktoken

from bot.api.models import ChatRequest, ChatResponse
from bot.chat import async_openai_client
from bot.graph.control_plane import ControlPlane
from bot.lang.parsers import extract_markdown_page_elements, get_html_soup, strip_html_elements
from bot.lang.patterns import naive_sentence_end_pattern
from bot.lang.tf_idf import get_tf_idf_keywords
from bot.websocket import (WebSocketDisconnectEventHandler, WebSocketReceiveEventHandler, WebSocketSendEventHandler,
                           WebSocketSender, WebSocketSessionMonitor)
from observability.annotations import measure_exec_seconds
from settings import germ_settings

logger = logging.getLogger(__name__)


class ChatMessageMeta:
    def __init__(self, user_id: int, conversation_id: int, dt_created: datetime,
                 content: str, token_size: int, vector_id: int, vector, emotions, keywords):
        self.content = content
        self.conversation_id = conversation_id
        self.dt_created = dt_created
        self.emotions = emotions
        self.keywords = keywords
        self.neighbors: list[int] = []
        self.token_size = token_size
        self.user_id = user_id
        self.vector = vector
        self.vector_id = vector_id


class ChatController(WebSocketDisconnectEventHandler, WebSocketReceiveEventHandler,
                     WebSocketSendEventHandler, WebSocketSessionMonitor):
    def __init__(
            self, control_plane: ControlPlane,
            delegate: WebSocketReceiveEventHandler,
            # Based on embeddings model
            embedding_dimensions: int = 3072,  # text-embedding-3-large can be shortened to 256
            token_model: str = "text-embedding-3-large",
            truncation_threshold: int = 8191,
    ):
        self.chat_messages: dict[str, ChatMessageMeta] = {}
        self.control_plane = control_plane
        self.conversations: dict[int, dict] = {}
        self.delegate = delegate
        self.embedding_dimensions: int = embedding_dimensions
        self.faiss_assistant_message = faiss.IndexIDMap(faiss.IndexFlatIP(embedding_dimensions))
        self.faiss_user_message = faiss.IndexIDMap(faiss.IndexFlatIP(embedding_dimensions))
        self.token_encoder = tiktoken.encoding_for_model(token_model)
        self.truncation_threshold = truncation_threshold

    async def get_text_embedding_vector(self, text: str):
        response = await async_openai_client.embeddings.create(
            dimensions=self.embedding_dimensions,
            model="text-embedding-3-large",
            input=text,
            encoding_format="float"
        )
        vector = np.array([response.data[0].embedding], dtype=np.float32)
        faiss.normalize_L2(vector)  # Important for cosine search
        return vector

    async def on_disconnect(self, conversation_id: int):
        pass

    @measure_exec_seconds(use_logging=True, use_prometheus=True)
    async def on_receive(self, user_id: int, conversation_id: int, dt_created: datetime, text_sig: str,
                         chat_request: ChatRequest, ws_sender: WebSocketSender):
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = {
                "est_token_size": 0,
                "messages": {},
            }
        dt_created_ts = int(dt_created.timestamp())
        self.conversations[conversation_id]["messages"][dt_created_ts] = {
            "user_id": user_id, "text_sig": text_sig
        }

        last_message_content = chat_request.messages[-1].content
        if text_sig not in self.chat_messages:
            embedding_vector_task = asyncio.create_task(
                self.get_text_embedding_vector(last_message_content)
            )

            est_token_size = len(self.token_encoder.encode(last_message_content))
            vector_id = convert_conversation_id_and_ts_to_vector_id(conversation_id, dt_created_ts)
            code_blocks, text_chunks, sentence_chunks = await process_markdown_elements(
                await run_in_threadpool(extract_markdown_page_elements, last_message_content)
            )

            embedding_vector, emotion_classification, tf_id_keywords = await asyncio.gather(*[
                embedding_vector_task,
                get_emotions_classifications(sentence_chunks),
                run_in_threadpool(get_tf_idf_keywords, sentence_chunks, top=5),
            ])
            self.chat_messages[text_sig] = meta = ChatMessageMeta(
                user_id=user_id, conversation_id=conversation_id, dt_created=dt_created,
                content=last_message_content, token_size=est_token_size,
                vector_id=vector_id, vector=embedding_vector,
                emotions=emotion_classification, keywords=tf_id_keywords
            )
        else:
            meta = self.chat_messages[text_sig]

        self.conversations[conversation_id]["est_token_size"] += meta.token_size
        logger.info(f"est. conversation tokens {self.conversations[conversation_id]['est_token_size']}")
        logger.info(f"embedding vector dims: {len(meta.vector[0])}")
        logger.info(f"emotions: {meta.emotions}")
        logger.info(f"tf-idf keywords: {meta.keywords}")

        # NOTES:
        # - Similar questions often recall better than responses and questions
        # - "tell me a joke" and "give me a side-slapper" mean the same thing, but they don't recall each other
        # - Even though ^ doesn't recall on the question, the responses are the same, and maybe we can link the
        #   questions based on close similarity of the responses
        # - Messages with a lot of links might be more valuable just like messages that gets recalled often. Maybe we
        #   can generate per sentence embeddings for these high value messages in cases there are higher value chunks
        #   in them.
        # - If we generate per sentence embeddings and link all sentences together in order, recalling any chunk would
        #   allow us to piece together an episode. This seems more like human episodic memory where an episode is not
        #   recalled in whole but in chunks.

        # Search for up to 4 items
        user_message_similarity_scores, user_message_neighbors = await run_in_threadpool(
            self.faiss_user_message.search, meta.vector, 4
        )
        for rank, (result_id, score) in enumerate(zip(user_message_neighbors[0], user_message_similarity_scores[0]), 1):
            if result_id != -1 and score > 0.7:  # -1 means no match
                result_conversation_id, result_message_ts = convert_vector_id_to_conversation_id_and_ts(result_id)
                if conversation_id != result_conversation_id:
                    result_sig = self.conversations[result_conversation_id]["messages"][result_message_ts]["text_sig"]
                    result_meta = self.chat_messages[result_sig]
                    logger.info(f"{rank:>2}. vector_id={result_id}  sim={score:.4f} text={result_meta.content}")

        assistant_message_similarity_scores, assistant_message_neighbors = await run_in_threadpool(
            self.faiss_assistant_message.search, meta.vector, 4
        )
        for rank, (result_id, score) in enumerate(zip(assistant_message_neighbors[0], assistant_message_similarity_scores[0]), 1):
            if result_id != -1 and score > 0.35:  # -1 means no match
                result_conversation_id, result_message_ts = convert_vector_id_to_conversation_id_and_ts(result_id)
                if conversation_id != result_conversation_id:
                    result_sig = self.conversations[result_conversation_id]["messages"][result_message_ts]["text_sig"]
                    result_meta = self.chat_messages[result_sig]
                    logger.info(f"{rank:>2}. vector_id={result_id}  sim={score:.4f} text={result_meta.content}")

        # Send to LLM
        await self.delegate.on_receive(user_id, conversation_id, dt_created, text_sig, chat_request, ws_sender)

    async def on_send(self, conversation_id: int, dt_created: datetime, text_sig: str,
                      chat_response: ChatResponse, received_message_dt_created: datetime = None):
        dt_created_ts = int(dt_created.timestamp())
        received_message_ts = int(received_message_dt_created.timestamp())
        received_message_sig = self.conversations[conversation_id]["messages"][received_message_ts]["text_sig"]
        received_message_meta = self.chat_messages[received_message_sig]
        self.conversations[conversation_id]["messages"][dt_created_ts] = {
            "user_id": 0, "text_sig": text_sig
        }

        if text_sig not in self.chat_messages:
            embedding_vector_task = asyncio.create_task(
                # TODO: Possibly replace code blocks
                self.get_text_embedding_vector(chat_response.content)
            )

            est_token_size = len(self.token_encoder.encode(chat_response.content))
            vector_id = convert_conversation_id_and_ts_to_vector_id(conversation_id, dt_created_ts)
            code_blocks, text_chunks, sentence_chunks = await process_markdown_elements(
                await run_in_threadpool(extract_markdown_page_elements, chat_response.content)
            )

            embedding_vector, tf_id_keywords = await asyncio.gather(*[
                embedding_vector_task,
                run_in_threadpool(get_tf_idf_keywords, ["\n".join(text_chunks)], top=15),
            ])
            self.chat_messages[text_sig] = meta = ChatMessageMeta(
                user_id=0, conversation_id=conversation_id, dt_created=dt_created,
                content=chat_response.content, token_size=est_token_size,
                vector_id=vector_id, vector=embedding_vector,
                emotions=None, keywords=tf_id_keywords
            )
        else:
            meta = self.chat_messages[text_sig]

        self.conversations[conversation_id]["est_token_size"] += meta.token_size
        logger.info(f"est. conversation tokens {self.conversations[conversation_id]['est_token_size']}")
        logger.info(f"embedding vector dims: {len(meta.vector[0])}")
        logger.info(f"tf-idf keywords: {meta.keywords}")

        await asyncio.gather(*[
            run_in_threadpool(
                self.faiss_user_message.add_with_ids,
                received_message_meta.vector,
                np.array([received_message_meta.vector_id], dtype=np.int64)
            ),
            run_in_threadpool(
                self.faiss_assistant_message.add_with_ids,
                embedding_vector,
                np.array([vector_id], dtype=np.int64)
            )
        ])

    async def on_tick(self, conversation_id: int, ws_sender: WebSocketSender):
        logger.info(f"conversation {conversation_id} is still active")


def convert_conversation_id_and_ts_to_vector_id(int_id: int, unix_ts: int) -> int:
    # 32-bit bitwise shifting - space efficient, invertible, non-commutative
    return ((int_id & 0xFFFFFFFF) << 32) | (unix_ts & 0xFFFFFFFF)


def convert_vector_id_to_conversation_id_and_ts(vector_id: int) -> tuple[int, int]:
    int_id = vector_id >> 32
    unix_ts = vector_id & 0xFFFFFFFF
    return int_id, unix_ts


async def get_emotions_classifications(texts: list[str]):
    async with aiohttp.ClientSession() as session:
        async with session.post(f"http://{germ_settings.MODEL_SERVICE_ENDPOINT}/text/classification/emotions",
                                json={"texts": texts}) as response:
            return await response.json()


async def process_markdown_elements(markdown_elements):
    code_blocks = []
    text_chunks = []
    sentence_chunks = []
    for element in markdown_elements:
        if element[0] in {"heading", "list_item", "paragraph"}:
            p_soup = await run_in_threadpool(get_html_soup, f"<p>{element[1]}</p>")
            p_text, p_elements = await strip_html_elements(p_soup, "p")

            text_chunks.append(p_text)
            text_chunk = p_text
            while text_chunk:
                # Not always perfect but good enough
                sentence_end_match = await run_in_threadpool(
                    naive_sentence_end_pattern.search, text_chunk, 0, len(text_chunk))
                if sentence_end_match:
                    sentence_chunks.append(text_chunk[:sentence_end_match.end()])
                    text_chunk = text_chunk[sentence_end_match.end():].strip()
                else:
                    sentence_chunks.append(text_chunk.strip())
                    text_chunk = ""
        elif element[0] == "block_code":
            code_blocks.append({"content": element[2], "language": element[1]})
        else:
            logger.info(f"skipped element type: {element[0]}")
    return code_blocks, text_chunks, sentence_chunks
