from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from starlette.concurrency import run_in_threadpool
import aiohttp
import asyncio
import logging
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
            token_model: str = "gpt-4",   # For estimation only since multiple models could be used
            truncation_threshold: int = 8192,  # Based on smaller GPT-4 variant
    ):
        self.control_plane = control_plane
        self.conversations: dict[int, dict] = {}
        self.delegate = delegate
        self.chunk_cache = {}
        self.message_cache = {}
        self.token_encoder = tiktoken.encoding_for_model(token_model)
        self.truncation_threshold = truncation_threshold

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
        self.conversations[conversation_id]["messages"][dt_created] = {
            "user_id": user_id, "text_sig": text_sig
        }

        last_message_content = chat_request.messages[-1].content
        if text_sig not in self.message_cache:
            est_token_size = len(self.token_encoder.encode(last_message_content))
            text_embeddings_task = asyncio.create_task(get_text_embeddings(last_message_content))

            code_blocks, text_chunks, sentence_chunks = await process_markdown_elements(
                await run_in_threadpool(extract_markdown_page_elements, last_message_content)
            )
            text_embeddings, emotion_classification, tf_id_keywords = await asyncio.gather(*[
                text_embeddings_task,
                get_emotions_classifications(sentence_chunks),
                get_tf_idf_keywords(sentence_chunks),
            ])
            self.message_cache[text_sig] = (
                last_message_content, est_token_size, text_embeddings, emotion_classification, tf_id_keywords
            )
        else:
            _, est_token_size, text_embeddings, emotion_classification, tf_id_keywords = self.message_cache[text_sig]

        self.conversations[conversation_id]["est_token_size"] += est_token_size
        logger.info(f"est. conversation tokens {self.conversations[conversation_id]['est_token_size']}")
        logger.info(f"text embeddings (10 of {len(text_embeddings)}): {text_embeddings[:10]}")
        logger.info(f"emotions: {emotion_classification}")
        logger.info(f"tf-idf keywords: {tf_id_keywords}")

        # Send to LLM
        await self.delegate.on_receive(user_id, conversation_id, dt_created, text_sig, chat_request, ws_sender)

    async def on_send(self, conversation_id: int, dt_created: datetime, text_sig: str,
                      chat_response: ChatResponse, received_message_dt_created: datetime = None):
        self.conversations[conversation_id]["messages"][dt_created] = {
            "user_id": 0, "text_sig": text_sig
        }

        if text_sig not in self.message_cache:
            est_token_size = len(self.token_encoder.encode(chat_response.content))
            text_embeddings_task = asyncio.create_task(get_text_embeddings(chat_response.content))

            code_blocks, text_chunks, sentence_chunks = await process_markdown_elements(
                await run_in_threadpool(extract_markdown_page_elements, chat_response.content)
            )
            text_embeddings, tf_id_keywords = await asyncio.gather(*[
                text_embeddings_task,
                get_tf_idf_keywords(sentence_chunks),
            ])
            self.message_cache[text_sig] = (
                chat_response.content, est_token_size, text_embeddings, None, tf_id_keywords
            )
        else:
            _, est_token_size, text_embeddings, _, tf_id_keywords = self.message_cache[text_sig]

        self.conversations[conversation_id]["est_token_size"] += est_token_size
        logger.info(f"est. conversation tokens {self.conversations[conversation_id]['est_token_size']}")
        logger.info(f"text embeddings (10 of {len(text_embeddings)}): {text_embeddings[:10]}")
        logger.info(f"tf-idf keywords: {tf_id_keywords}")

    async def on_tick(self, conversation_id: int, ws_sender: WebSocketSender):
        logger.info(f"conversation {conversation_id} is still active")


async def get_emotions_classifications(texts: list[str]):
    async with aiohttp.ClientSession() as session:
        async with session.post(f"http://{germ_settings.MODEL_SERVICE_ENDPOINT}/text/classification/emotions",
                                json={"texts": texts}) as response:
            return await response.json()


async def get_text_embeddings(text: str):
    response = await async_openai_client.embeddings.create(
        model="text-embedding-3-large",
        input=text,
        encoding_format="float"
    )
    return response.data[0].embedding


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
