from sklearn.feature_extraction.text import TfidfVectorizer
from starlette.concurrency import run_in_threadpool
import asyncio
import logging

from bot.api.models import ChatRequest, ChatResponse
from bot.graph.control_plane import ControlPlane
from bot.websocket import (WebSocketDisconnectEventHandler, WebSocketReceiveEventHandler, WebSocketSendEventHandler,
                           WebSocketSender, WebSocketSessionMonitor)
from observability.annotations import measure_exec_seconds

logger = logging.getLogger(__name__)

tf_idf_vectorizer = TfidfVectorizer(stop_words='english')


class ChatController(WebSocketDisconnectEventHandler, WebSocketReceiveEventHandler,
                     WebSocketSendEventHandler, WebSocketSessionMonitor):

    def __init__(self, control_plane: ControlPlane, remote: WebSocketReceiveEventHandler):
        self.control_plane = control_plane
        self.remote = remote

    async def get_tf_idf_keywords(self, text: str):
        documents = [text]
        # The IDF part means I should retrieve a bunch of random documents from memory and append them to documents.
        tfidf_matrix = await run_in_threadpool(tf_idf_vectorizer.fit_transform, documents, y=None)
        feature_names = tf_idf_vectorizer.get_feature_names_out()
        first_doc_scores = tfidf_matrix[0].toarray().flatten()
        top_indices = first_doc_scores.argsort()[::-1]
        top_terms = [feature_names[i] for i in top_indices[:5]]
        logger.info(f"Top terms: {top_terms}")

    async def on_disconnect(self, conversation_id: int):
        pass

    @measure_exec_seconds(use_logging=True, use_prometheus=True)
    async def on_receive(self, conversation_id: int, chat_request_received_id: int, chat_request: ChatRequest,
                         ws_sender: WebSocketSender):
        remote_response_task = asyncio.create_task(
            self.remote.on_receive(conversation_id, chat_request_received_id, chat_request, ws_sender),
        )
        # TODO: Do stuff.
        tf_id_keywords = await self.get_tf_idf_keywords(chat_request.messages[-1].content)

        await remote_response_task

    async def on_send(self,
                      sent_message_id: int,
                      chat_response: ChatResponse,
                      conversation_id: int,
                      received_message_id: int = None):
        tf_id_keywords = await self.get_tf_idf_keywords(chat_response.content)

    async def on_tick(self, conversation_id: int, ws_sender: WebSocketSender):
        logger.info(f"conversation {conversation_id} is still active")
