import logging
from datetime import datetime

from germ.api.models import ChatRequest, ChatResponse
from germ.services.bot.chat import async_openai_client
from germ.services.bot.websocket import WebSocketReceiveEventHandler, WebSocketSender
from germ.observability.annotations import measure_exec_seconds
from germ.settings import germ_settings

logger = logging.getLogger(__name__)


class ChatModelEventHandler(WebSocketReceiveEventHandler):
    def __init__(self, httpx_timeout: float = 30):
        self.httpx_timeout = httpx_timeout

    @measure_exec_seconds(use_logging=True, use_prometheus=True)
    async def on_receive(
            self, user_id: int, conversation_id: int, dt_created: datetime,
            chat_request: ChatRequest, ws_sender: WebSocketSender
    ):
        if chat_request.model is None:
            chat_request.model = germ_settings.CHAT_MODEL
        if chat_request.timeout is None:
            chat_request.timeout = self.httpx_timeout

        completion = await async_openai_client.chat.completions.create(
            messages=[message.model_dump() for message in chat_request.messages] + [
                {"role": "system",
                 "content": ("Answer in valid Markdown format only. "
                             "Don't use code blocks unnecessarily but always use them when dealing with code.")}
            ],
            model=chat_request.model,
            timeout=chat_request.timeout,
        )

        await ws_sender.send_reply(
            dt_created,
            ChatResponse(
                complete=True,
                content=completion.choices[0].message.content,
                model=completion.model
            )
        )


class ReasoningChatModelEventHandler(WebSocketReceiveEventHandler):
    def __init__(self, httpx_timeout: float = 180):
        self.httpx_timeout = httpx_timeout

    @measure_exec_seconds(use_logging=True, use_prometheus=True)
    async def on_receive(
            self, user_id: int, conversation_id: int, dt_created: datetime,
            chat_request: ChatRequest, ws_sender: WebSocketSender
    ):
        if chat_request.model is None:
            chat_request.model = germ_settings.REASONING_MODEL
        if chat_request.reasoning_effort is None:
            chat_request.reasoning_effort = "low"
        if chat_request.timeout is None:
            chat_request.timeout = self.httpx_timeout

        completion = await async_openai_client.chat.completions.create(
            messages=[message.model_dump() for message in chat_request.messages] + [
                {"role": "system",
                 "content": ("Answer in valid Markdown format only. "
                             "Don't use code blocks unnecessarily but always use them when dealing with code.")}
            ],
            model=chat_request.model,
            reasoning_effort=chat_request.reasoning_effort,
            timeout=chat_request.timeout,
        )

        await ws_sender.send_reply(
            dt_created,
            ChatResponse(
                complete=True,
                content=completion.choices[0].message.content,
                model=f"{completion.model}[effort:{chat_request.reasoning_effort}]"
            )
        )
