from starlette.concurrency import run_in_threadpool
import aiohttp
import asyncio
import logging

from bot.api.models import ChatRequest, TextPayload
from bot.lang.parsers import extract_markdown_page_elements
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
        # TODO:
        #  - implement local completions
        #  - maybe even user profiling should be moved here...
        elements = await run_in_threadpool(extract_markdown_page_elements, chat_request.messages[-1].content)
        for element in elements:
            if element[0] == "paragraph":
                async with aiohttp.ClientSession() as session:
                    async with session.post("http://germ-models:9000/text/classification",
                                            json={"text": element[1]}) as response:
                        data = await response.json()
                        logger.info("token classifications:\n" + (
                            "\n".join([f"{head}\t{labels}" for head, labels in data.items()])))
            else:
                logger.info(f"ignored element type: {element[0]}")
                continue

        await remote_response_task
