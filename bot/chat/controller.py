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
        self.remote = remote

    async def on_disconnect(self, conversation_id: int):
        pass

    @measure_exec_seconds(use_logging=True, use_prometheus=True)
    async def on_receive(self, conversation_id: int, chat_request_received_id: int, chat_request: ChatRequest,
                         ws_sender: WebSocketSender):
        remote_response_task = asyncio.create_task(
            self.remote.on_receive(conversation_id, chat_request_received_id, chat_request, ws_sender),
        )
        # TODO: Do stuff.
        await remote_response_task

    async def on_send(self,
                      sent_message_id: int,
                      chat_response: ChatResponse,
                      conversation_id: int,
                      received_message_id: int = None):
        pass

    async def on_tick(self, conversation_id: int, ws_sender: WebSocketSender):
        logger.info(f"conversation {conversation_id} is still active")
