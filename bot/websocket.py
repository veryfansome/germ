from abc import ABC, abstractmethod
from fastapi import WebSocket
from openai import OpenAI
from prometheus_client import Gauge
from sqlalchemy.sql import desc
from starlette.concurrency import run_in_threadpool
import asyncio
import datetime
import logging

from api.models import ChatMessage, ChatRequest, ChatResponse, ChatSessionSummary
from settings.openai_settings import DEFAULT_CHAT_MODEL
from db.models import ChatSession, ChatRequestReceived, ChatResponseSent, SessionLocal

logger = logging.getLogger(__name__)

##
# Metrics

METRIC_CHAT_SESSIONS_IN_PROGRESS = Gauge(
    "chat_sessions_in_progress", "Number of connected sessions that have yet to be disconnected")


class WebSocketSender:
    def __init__(self, chat_session_id: int, chat_request_received_id: int, connection: WebSocket):
        self.chat_session_id = chat_session_id
        self.chat_request_received_id = chat_request_received_id
        self.connection = connection

    async def send_chat_response(self, chat_response: ChatResponse):
        await self.connection.send_text(chat_response.model_dump_json())
        new_chat_response_sent(self.chat_session_id, self.chat_request_received_id, chat_response)


class WebSocketEventHandler(ABC):
    @abstractmethod
    async def on_receive(self,
                         chat_session_id: int, chat_request_received_id: id,
                         chat_request: ChatRequest, response_sender: WebSocketSender):
        pass


class WebSocketConnectionManager:
    def __init__(self):
        self.active_connections: dict[int, WebSocket] = {}
        self.event_handlers: list[WebSocketEventHandler] = []

    def add_event_handler(self, handler: WebSocketEventHandler):
        self.event_handlers.append(handler)

    async def broadcast(self, message: str):
        for connection in self.active_connections.values():
            await connection.send_text(message)

    async def connect(self, websocket: WebSocket) -> int:
        chat_session_id = await run_in_threadpool(new_chat_session)
        await websocket.accept()
        self.active_connections[chat_session_id] = websocket
        METRIC_CHAT_SESSIONS_IN_PROGRESS.inc()
        return chat_session_id

    async def disconnect(self, chat_session_id: int):
        async def runnable():
            update_chat_session_time_stopped(chat_session_id)
            messages = await run_in_threadpool(get_chat_session_messages, chat_session_id)
            if len(messages) > 2:  # A conversation should have at least a message and a reply
                with OpenAI() as client:
                    completion = client.chat.completions.create(
                        messages=([m.model_dump() for m in messages] + [{
                            "role": "user", "content": "Give this conversation a title using 10 or fewer words."
                        }]),
                        model=DEFAULT_CHAT_MODEL, n=1,
                        temperature=0.0,
                    )
                    update_chat_session_summary(chat_session_id, completion.choices[0].message.content)
            else:
                logger.info("skipped adding chat session summary")
        await asyncio.create_task(runnable())
        self.active_connections.pop(chat_session_id, None)
        METRIC_CHAT_SESSIONS_IN_PROGRESS.dec()

    async def disconnect_all(self):
        for chat_session_id in self.active_connections:
            async def runnable():
                await self.disconnect(chat_session_id)
            await asyncio.create_task(runnable())

    async def receive(self, chat_session_id: int):
        connection: WebSocket = self.active_connections[chat_session_id]
        chat_request = ChatRequest.parse_obj(await connection.receive_json())
        chat_request_received_id = await run_in_threadpool(new_chat_request_received,
                                                           chat_session_id, chat_request)
        response_sender = WebSocketSender(chat_session_id, chat_request_received_id, connection)
        for handler in self.event_handlers:
            async def runnable():
                await handler.on_receive(chat_session_id, chat_request_received_id, chat_request, response_sender)
            await asyncio.create_task(runnable())


def get_chat_session_messages(chat_session_id: int) -> list[ChatMessage]:
    messages: list[ChatMessage] = []
    with SessionLocal() as session:
        response_sent = session.query(ChatResponseSent).filter_by(
            chat_session_id=chat_session_id
        ).order_by(
            desc(ChatResponseSent.chat_response_sent_id)
        ).first()
        if response_sent:
            request_received = session.query(ChatRequestReceived).join(ChatResponseSent).filter(
                ChatRequestReceived.chat_request_received_id == ChatResponseSent.chat_request_received_id,
                ChatResponseSent.chat_response_sent_id == response_sent.chat_response_sent_id
            ).one_or_none()
            if request_received:
                for m in ChatRequest.parse_obj(request_received.chat_request).messages:
                    messages.append(m)
            messages.append(
                ChatMessage(
                    role="assistant",
                    content=ChatResponse.parse_obj(response_sent.chat_response).response.choices[0].message.content))
    return messages


def get_chat_session_summaries() -> list[ChatSessionSummary]:
    cs_list = []
    with SessionLocal() as session:
        results = session.query(ChatSession).filter(
            ChatSession.is_hidden.is_(False),
            ChatSession.summary.is_not(None),
            ChatSession.time_stopped.is_not(None),
        ).all()
        for cs in results:
            cs_list.append(ChatSessionSummary(
                chat_session_id=cs.chat_session_id,
                summary=cs.summary,
                time_started=cs.time_started,
                time_stopped=cs.time_stopped,
            ))
    return cs_list


def new_chat_request_received(chat_session_id: int, chat_request: ChatRequest) -> int:
    with SessionLocal() as session:
        stored_request = ChatRequestReceived(
            chat_session_id=chat_session_id,
            chat_request=chat_request.model_dump(),
        )
        session.add(stored_request)
        session.commit()
        return stored_request.chat_request_received_id


def new_chat_response_sent(chat_session_id: int, chat_request_received_id: int, chat_response: ChatResponse) -> int:
    with SessionLocal() as session:
        stored_response = ChatResponseSent(
            chat_session_id=chat_session_id,
            chat_request_received_id=chat_request_received_id,
            chat_response=chat_response.model_dump(),
        )
        session.add(stored_response)
        session.commit()
        return stored_response.chat_response_sent_id


def new_chat_session() -> int:
    with SessionLocal() as session:
        cs = ChatSession()
        session.add(cs)
        session.commit()
        return cs.chat_session_id


def update_chat_session_is_hidden(chat_session_id: int):
    with SessionLocal() as session:
        cs = session.query(ChatSession).filter_by(chat_session_id=chat_session_id).first()
        if cs:
            cs.is_hidden = True
            session.commit()
        else:
            logger.error(f"ChatSession.chat_session_id == {chat_session_id} not found")


def update_chat_session_summary(chat_session_id: int, summary: str):
    with SessionLocal() as session:
        cs = session.query(ChatSession).filter_by(chat_session_id=chat_session_id).first()
        if cs:
            cs.summary = summary
            session.commit()
        else:
            logger.error(f"ChatSession.chat_session_id == {chat_session_id} not found")


def update_chat_session_time_stopped(chat_session_id: int):
    with SessionLocal() as session:
        cs = session.query(ChatSession).filter_by(chat_session_id=chat_session_id).first()
        if cs:
            cs.time_stopped = datetime.datetime.now(datetime.timezone.utc)
            session.commit()
        else:
            logger.error(f"ChatSession.chat_session_id == {chat_session_id} not found")
