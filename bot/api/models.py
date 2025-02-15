from datetime import datetime
from pydantic import BaseModel
from typing import Any, Optional


class ChatBookmark(BaseModel):
    id: Optional[int] = None
    is_test: Optional[bool] = True
    message_received_id: Optional[int] = None
    message_replied_id: Optional[int] = None
    message_summary: Optional[str] = None


class ChatMessage(BaseModel):
    content: str
    role: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    parameters: Optional[dict[str, str]] = None
    uploaded_filenames: Optional[list[Any]] = None


class ChatResponse(ChatMessage):
    complete: Optional[bool] = None
    model: str = "none"
    role: str = "assistant"


class ChatSessionSummary(BaseModel):
    chat_session_id: Optional[int] = None
    summary: Optional[str] = None
    time_started: Optional[datetime] = None
    time_stopped: Optional[datetime] = None


class SqlRequest(BaseModel):
    sql: str
