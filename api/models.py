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


class ChatThumbsDown(BaseModel):
    id: Optional[int] = None
    is_test: Optional[bool] = True
    message_received_id: Optional[int] = None
    message_replied_id: Optional[int] = None


class SqlRequest(BaseModel):
    sql: str
