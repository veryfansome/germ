import asyncio
import logging

from bot.api.models import ChatRequest
from bot.websocket import WebSocketReceiveEventHandler, WebSocketSender
from observability.annotations import measure_exec_seconds

logger = logging.getLogger(__name__)


class ChatController(WebSocketReceiveEventHandler):

    def __init__(self, remote: WebSocketReceiveEventHandler):
        self.remote = remote

    @measure_exec_seconds(use_logging=True, use_prometheus=True)
    async def on_receive(self, chat_session_id: int, chat_request_received_id: int, chat_request: ChatRequest,
                         ws_sender: WebSocketSender):
        remote_response_task = asyncio.create_task(
            self.remote.on_receive(chat_session_id, chat_request_received_id, chat_request, ws_sender),
        )
        # TODO: implement local completions

        await remote_response_task
