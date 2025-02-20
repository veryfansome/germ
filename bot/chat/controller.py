import asyncio
import logging

from bot.api.models import ChatRequest, ChatResponse
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
        local_responses = [
            ChatResponse(complete=True, content="TBD", model="germ")
        ]

        remote_responses = await remote_response_task
        logger.info(f"choosing between {local_responses} and {remote_responses}")
        # False is a signal to not respond, e.g. file uploads \w responses via threads
        if remote_responses is not False:
            # TODO: Chose either local or a remote response
            if remote_responses is not None:
                for response in remote_responses:
                    _ = asyncio.create_task(ws_sender.return_chat_response(chat_request_received_id, response))
