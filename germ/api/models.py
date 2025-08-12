from pydantic import BaseModel
from typing import Any


class ChatMessage(BaseModel):
    content: str
    role: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    model: str | None = None
    parameters: dict[str, str] | None = None
    reasoning_effort: str | None = None
    timeout: float | None = None
    uploaded_filenames: list[Any] | None = None


class ChatResponse(ChatMessage):
    complete: bool | None = None
    conversation_ident: str | None = None
    error: bool | None = False
    model: str = "none"
    role: str = "assistant"


class TextListPayload(BaseModel):
    texts: list[str]


class EmbeddingRequestPayload(TextListPayload):
    prompt: str = "query: "
