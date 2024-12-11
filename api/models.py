from datetime import datetime
from pydantic import BaseModel
from typing import Optional


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
    system_message: Optional[str] = ""
    temperature: Optional[float] = 0.0


class ChatResponse(ChatMessage):
    role: str = "assistant"
    model: str = "none"


class ChatSessionSummary(BaseModel):
    chat_session_id: Optional[int] = None
    summary: Optional[str] = None
    time_started: Optional[datetime] = None
    time_stopped: Optional[datetime] = None


class SqlRequest(BaseModel):
    sql: str
