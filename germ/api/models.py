from pydantic import BaseModel
from typing import Any, Optional


class ChatMessage(BaseModel):
    content: str
    role: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    parameters: Optional[dict[str, str]] = None
    uploaded_filenames: Optional[list[Any]] = None


class ChatResponse(ChatMessage):
    complete: Optional[bool] = None
    conversation_ident: Optional[str] = None
    error: Optional[bool] = False
    model: str = "none"
    role: str = "assistant"


class TextListPayload(BaseModel):
    texts: list[str]


class EmbeddingRequestPayload(TextListPayload):
    prompt: str = "query: "
