from sklearn.feature_extraction.text import TfidfVectorizer
from starlette.concurrency import run_in_threadpool
import aiohttp
import asyncio
import logging

from bot.api.models import ChatRequest, ChatResponse
from bot.chat import async_openai_client
from bot.graph.control_plane import ControlPlane
from bot.lang.parsers import extract_markdown_page_elements
from bot.lang.patterns import naive_sentence_end_pattern
from bot.websocket import (InterceptingWebSocketSender,
                           WebSocketDisconnectEventHandler, WebSocketReceiveEventHandler, WebSocketSendEventHandler,
                           WebSocketSender, WebSocketSessionMonitor)
from observability.annotations import measure_exec_seconds

logger = logging.getLogger(__name__)

tf_idf_vectorizer = TfidfVectorizer(stop_words='english')


class ChatController(WebSocketDisconnectEventHandler, WebSocketReceiveEventHandler,
                     WebSocketSendEventHandler, WebSocketSessionMonitor):
    def __init__(self, control_plane: ControlPlane, delegate: WebSocketReceiveEventHandler):
        self.control_plane = control_plane
        self.delegate = delegate

    async def on_disconnect(self, conversation_id: int):
        pass

    @measure_exec_seconds(use_logging=True, use_prometheus=True)
    async def on_receive(self, conversation_id: int, chat_request_received_id: int, chat_request: ChatRequest,
                         ws_sender: WebSocketSender):
        interceptor = InterceptingWebSocketSender(ws_sender)  # Intercepts and holds LLM response
        fast_response_task = asyncio.create_task(
            # Send to LLM
            self.delegate.on_receive(conversation_id, chat_request_received_id, chat_request, interceptor),
        )
        text_embeddings_task = asyncio.create_task(get_text_embeddings(chat_request.messages[-1].content))

        # Analyze most recent message in chat request
        code_blocks, text_chunks = await process_markdown_elements(
            await run_in_threadpool(extract_markdown_page_elements, chat_request.messages[-1].content)
        )
        sentence_chunks = []
        for text_chunk in text_chunks:
            while text_chunk:
                # Not always perfect but good enough
                sentence_end_match = await run_in_threadpool(
                    naive_sentence_end_pattern.search, text_chunk, 0, len(text_chunk))
                if sentence_end_match:
                    sentence_chunks.append(text_chunk[:sentence_end_match.end()])
                    text_chunk = text_chunk[sentence_end_match.end():].strip()
                else:
                    sentence_chunks.append(text_chunk)
                    text_chunk = ""

        embeddings, emotion_classification, tf_id_keywords = await asyncio.gather(*[
            text_embeddings_task,
            get_emotions_classifications(sentence_chunks),
            get_tf_idf_keywords(sentence_chunks),
        ])
        logger.info(f"embeddings (10 of {len(embeddings)}): {embeddings[:10]}")
        logger.info(f"emotions: {emotion_classification}")
        logger.info(f"tf-idf keywords: {tf_id_keywords}")
        # TODO:
        #  - Determine if the intercepted initial fast response is good enough or if a new response should
        #    be generated.
        await fast_response_task
        await interceptor.send_intercepted_responses()

    async def on_send(self,
                      sent_message_id: int,
                      chat_response: ChatResponse,
                      conversation_id: int,
                      received_message_id: int = None):
        pass

    async def on_tick(self, conversation_id: int, ws_sender: WebSocketSender):
        logger.info(f"conversation {conversation_id} is still active")


async def get_emotions_classifications(texts: list[str]):
    async with aiohttp.ClientSession() as session:
        async with session.post("http://germ-models:9000/text/classification/emotions",
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
    for element in markdown_elements:
        if element[0] in {"heading", "list_item", "paragraph"}:
            text_chunks.append(element[1])
        elif element[0] == "block_code":
            code_blocks.append({"content": element[2], "language": element[1]})
        else:
            logger.info(f"skipped element type: {element[0]}")
    return code_blocks, text_chunks
