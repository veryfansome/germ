from datetime import datetime
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
    conversation_id: Optional[int] = None
    model: str = "none"
    role: str = "assistant"


class TextPayload(BaseModel):
    text: str
