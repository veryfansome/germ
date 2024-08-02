from openai import OpenAI
from openai.types.chat.chat_completion import ChatCompletion
from starlette.concurrency import run_in_threadpool

from api.models import ChatRequest, ChatResponse
from bot.websocket import WebSocketEventHandler, WebSocketSender

ENABLED_CHAT_MODELS = (
    "gpt-4o-mini",
    "gpt-4o",
)
ENABLED_IMAGE_MODELS = (
)
CHAT_HANDLERS = {}


class OpenAIChatModelEventHandler(WebSocketEventHandler):
    def __init__(self, model):
        self.model = model

    async def on_receive(self,
                         chat_session_id: int, chat_request_received_id: int,
                         chat_request: ChatRequest, response_sender: WebSocketSender):
        completion = await run_in_threadpool(self.do_chat_completion, chat_request)
        await response_sender.send_chat_response(ChatResponse(response=completion))

    def do_chat_completion(self, chat_request: ChatRequest) -> ChatCompletion:
        with OpenAI() as client:
            return client.chat.completions.create(
                messages=[message.dict() for message in chat_request.messages],
                model=self.model,
                n=1,
                temperature=chat_request.temperature,
            )


for m in ENABLED_CHAT_MODELS:
    CHAT_HANDLERS[m] = OpenAIChatModelEventHandler(m)
