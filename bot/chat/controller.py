import asyncio
import logging

from bot.api.models import ChatRequest, ChatResponse
from bot.graph.control_plane import ControlPlane
from bot.websocket import (WebSocketDisconnectEventHandler, WebSocketReceiveEventHandler, WebSocketSendEventHandler,
                           WebSocketSender, WebSocketSessionMonitor)
from observability.annotations import measure_exec_seconds

logger = logging.getLogger(__name__)


class ChatController(WebSocketDisconnectEventHandler, WebSocketReceiveEventHandler,
                     WebSocketSendEventHandler, WebSocketSessionMonitor):

    def __init__(self, control_plane: ControlPlane, remote: WebSocketReceiveEventHandler):
        self.control_plane = control_plane
        self.node_types = {
            "block_code": "CodeBlock",
            "list": "Paragraph",
            "paragraph": "Paragraph",
        }
        self.remote = remote

    async def on_disconnect(self, chat_session_id: int):
        pass

    @measure_exec_seconds(use_logging=True, use_prometheus=True)
    async def on_receive(self, chat_session_id: int, chat_request_received_id: int, chat_request: ChatRequest,
                         ws_sender: WebSocketSender):
        remote_response_task = asyncio.create_task(
            self.remote.on_receive(chat_session_id, chat_request_received_id, chat_request, ws_sender),
        )
        # TODO: Do stuff.
        await remote_response_task

    async def on_send(self,
                      chat_response_sent_id: int,
                      chat_response: ChatResponse,
                      chat_session_id: int,
                      chat_request_received_id: int = None):
        pass

    async def on_tick(self, chat_session_id: int, ws_sender: WebSocketSender):
        logger.info(f"chat session {chat_session_id} is still active")
