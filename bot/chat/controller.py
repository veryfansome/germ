from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
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
from bot.websocket import (WebSocketDisconnectEventHandler, WebSocketReceiveEventHandler, WebSocketSendEventHandler,
                           WebSocketSender, WebSocketSessionMonitor)
from observability.annotations import measure_exec_seconds
from settings import germ_settings

logger = logging.getLogger(__name__)

tf_idf_vectorizer = TfidfVectorizer(stop_words='english')


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
        self.control_plane = control_plane
        self.conversations: dict[int, dict] = {}
        self.delegate = delegate
        self.embedding_dimensions: int = embedding_dimensions
        self.faiss_index_id_map = faiss.IndexIDMap(faiss.IndexFlatIP(embedding_dimensions))
        self.message_cache = {}
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
        if text_sig not in self.message_cache:
            est_token_size = len(self.token_encoder.encode(last_message_content))
            embedding_vector_task = asyncio.create_task(self.get_text_embedding_vector(last_message_content))
            vector_id = convert_conversation_id_and_ts_to_vector_id(conversation_id, dt_created_ts)

            code_blocks, text_chunks, sentence_chunks = await process_markdown_elements(
                await run_in_threadpool(extract_markdown_page_elements, last_message_content)
            )
            embedding_vector, emotion_classification, tf_id_keywords = await asyncio.gather(*[
                embedding_vector_task,
                get_emotions_classifications(sentence_chunks),
                get_tf_idf_keywords(sentence_chunks),
            ])
            self.message_cache[text_sig] = (
                last_message_content, est_token_size, vector_id, embedding_vector, emotion_classification, tf_id_keywords
            )
        else:
            _, est_token_size, vector_id, embedding_vector, emotion_classification, tf_id_keywords = self.message_cache[text_sig]

        self.conversations[conversation_id]["est_token_size"] += est_token_size
        logger.info(f"est. conversation tokens {self.conversations[conversation_id]['est_token_size']}")
        logger.info(f"embedding vector ({len(embedding_vector[0])}): {embedding_vector[0]}")
        logger.info(f"emotions: {emotion_classification}")
        logger.info(f"tf-idf keywords: {tf_id_keywords}")

        similarity_scores, neighbors = await run_in_threadpool(self.faiss_index_id_map.search, embedding_vector, 2)
        for rank, (result_id, score) in enumerate(zip(neighbors[0], similarity_scores[0]), 1):
            if result_id != -1:  # -1 means no match
                result_conversation_id, result_message_ts = convert_vector_id_to_conversation_id_and_ts(result_id)
                result_text = self.message_cache[self.conversations[result_conversation_id]["messages"][result_message_ts]["text_sig"]][0]
                logger.info(f"{rank:>2}. vector_id={result_id}  sim={score:.4f} text={result_text}")

        # Send to LLM
        await self.delegate.on_receive(user_id, conversation_id, dt_created, text_sig, chat_request, ws_sender)

    async def on_send(self, conversation_id: int, dt_created: datetime, text_sig: str,
                      chat_response: ChatResponse, received_message_dt_created: datetime = None):
        dt_created_ts = int(dt_created.timestamp())
        self.conversations[conversation_id]["messages"][dt_created_ts] = {
            "user_id": 0, "text_sig": text_sig
        }

        if text_sig not in self.message_cache:
            est_token_size = len(self.token_encoder.encode(chat_response.content))
            embedding_vector_task = asyncio.create_task(self.get_text_embedding_vector(chat_response.content))
            vector_id = convert_conversation_id_and_ts_to_vector_id(conversation_id, dt_created_ts)

            code_blocks, text_chunks, sentence_chunks = await process_markdown_elements(
                await run_in_threadpool(extract_markdown_page_elements, chat_response.content)
            )
            embedding_vector, tf_id_keywords = await asyncio.gather(*[
                embedding_vector_task,
                get_tf_idf_keywords(sentence_chunks),
            ])
            self.message_cache[text_sig] = (
                chat_response.content, est_token_size, vector_id, embedding_vector, None, tf_id_keywords
            )
        else:
            _, est_token_size, vector_id, embedding_vector, _, tf_id_keywords = self.message_cache[text_sig]

        self.conversations[conversation_id]["est_token_size"] += est_token_size
        logger.info(f"est. conversation tokens {self.conversations[conversation_id]['est_token_size']}")
        logger.info(f"embedding vector ({len(embedding_vector[0])}): {embedding_vector[0]}")
        logger.info(f"tf-idf keywords: {tf_id_keywords}")

        await run_in_threadpool(
            self.faiss_index_id_map.add_with_ids,
            embedding_vector, np.array([vector_id], dtype=np.int64)
        )

    async def on_tick(self, conversation_id: int, ws_sender: WebSocketSender):
        logger.info(f"conversation {conversation_id} is still active")


# TODO: can collide once conversation_id catches up to timestamp in magnitude
def convert_conversation_id_and_ts_to_vector_id(int_id: int, unix_ts: int) -> int:
    # Mask both values to 32 bits
    int_id_32 = int_id & 0xFFFFFFFF
    unix_ts_32 = unix_ts & 0xFFFFFFFF

    # Shift the timestamp by 32 bits and combine
    return (unix_ts_32 << 32) | int_id_32


def convert_vector_id_to_conversation_id_and_ts(vector_id: int) -> tuple[int, int]:
    # Extract the int ID (lower 32 bits)
    int_id = vector_id & 0xFFFFFFFF
    # Extract the timestamp (upper 32 bits)
    unix_ts = (vector_id >> 32) & 0xFFFFFFFF

    # Optionally convert back to Python's signed integers if needed
    return int_id, unix_ts


async def get_emotions_classifications(texts: list[str]):
    async with aiohttp.ClientSession() as session:
        async with session.post(f"http://{germ_settings.MODEL_SERVICE_ENDPOINT}/text/classification/emotions",
                                json={"texts": texts}) as response:
            return await response.json()


async def get_tf_idf_keywords(texts: list[str], top: int = 3):
    keywords = []
    documents = texts
    # The IDF part means I more documents with a broader blend of keywords.

    try:
        tfidf_matrix = await run_in_threadpool(tf_idf_vectorizer.fit_transform, documents, y=None)
        feature_names = tf_idf_vectorizer.get_feature_names_out()
        for text_idx, text in enumerate(texts):
            doc_scores = tfidf_matrix[text_idx].toarray().flatten()
            top_indices = doc_scores.argsort()[::-1]
            keywords.append({"text": text, "keywords": [feature_names[i] for i in top_indices[:top]]})
    except Exception as e:
        logger.error(f"Exception: {e}")
    return keywords


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
