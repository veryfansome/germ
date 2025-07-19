from pydantic import BaseModel

from germ.api.models import ChatRequest
from germ.services.bot.chat import async_openai_client


class ChatRequestClassification(BaseModel):
    foo: str | None = None


class ChatRequestClassifier:

    @classmethod
    async def classify(cls, chat_request: ChatRequest) -> ChatRequestClassification:
        return ChatRequestClassification()