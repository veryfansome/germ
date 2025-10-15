import logging
from datetime import datetime

from germ.api.models import ChatRequest, ChatResponse
from germ.services.bot.chat import async_openai_client
from germ.services.bot.websocket import WebSocketReceiveEventHandler, WebSocketSender
from germ.observability.annotations import measure_exec_seconds
from germ.settings import germ_settings

logger = logging.getLogger(__name__)


class ChatModelEventHandler(WebSocketReceiveEventHandler):
    def __init__(self, httpx_timeout: float = 30.0):
        self.httpx_timeout = httpx_timeout

    @measure_exec_seconds(use_logging=True, use_prometheus=True)
    async def on_receive(
            self, session, conversation_id: int, dt_created: datetime,
            chat_request: ChatRequest, ws_sender: WebSocketSender
    ):
        if chat_request.model is None:
            chat_request.model = germ_settings.OPENAI_CHAT_MODEL
        if chat_request.timeout is None:
            chat_request.timeout = self.httpx_timeout

        completion = await async_openai_client.chat.completions.create(
            messages=[message.model_dump() for message in chat_request.messages] + [get_system_message()],
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
    def __init__(self, httpx_timeout: float = 180.0):
        self.httpx_timeout = httpx_timeout

    @measure_exec_seconds(use_logging=True, use_prometheus=True)
    async def on_receive(
            self, session, conversation_id: int, dt_created: datetime,
            chat_request: ChatRequest, ws_sender: WebSocketSender
    ):
        if chat_request.model is None:
            chat_request.model = germ_settings.OPENAI_REASONING_MODEL
        if chat_request.reasoning_effort is None:
            chat_request.reasoning_effort = "low"
        if chat_request.timeout is None:
            chat_request.timeout = self.httpx_timeout

        completion = await async_openai_client.chat.completions.create(
            messages=[message.model_dump() for message in chat_request.messages] + [get_system_message()],
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


def get_system_message() -> dict[str, str]:
    return {"role": "system",
     "content": """
Formatting:
- Return valid Markdown.
- For inline code or when mentioning literal Markdown/HTML characters or tags, wrap in single backticks.
- For code blocks, use fenced code with triple backticks and include a language identifier when applicable.
- For callouts (notes, tips, warnings), use Markdown blockquotes with >.
""".strip()
     }