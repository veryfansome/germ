import json

from fastapi import WebSocket
from prometheus_client import Gauge
from starlette.concurrency import run_in_threadpool
import asyncio
import datetime
import logging

from api.models import ChatMessage, ChatRequest
from bot.v1 import chat as v1_chat
from bot.v2 import chat as v2_chat
from db.models import SessionLocal, ChatSession

logger = logging.getLogger(__name__)

##
# Metrics

METRIC_CHAT_SESSIONS_IN_PROGRESS = Gauge(
    "chat_sessions_in_progress", "Number of connected sessions that have yet to be disconnected")


class WebSocketOnReceiveEventHandler:
    pass


class WebSocketConnectionManager:
    def __init__(self):
        self.active_connections: dict[int, WebSocket] = {}
        self.on_receive_event_handlers: list[WebSocketOnReceiveEventHandler] = []

    def add_on_receive_event_handler(self, handler: WebSocketOnReceiveEventHandler):
        self.on_receive_event_handlers.append(handler)

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
        async def mark_time_stopped():
            mark_chat_session_time_stopped(chat_session_id)
        await asyncio.create_task(mark_time_stopped())
        self.active_connections.pop(chat_session_id, None)
        METRIC_CHAT_SESSIONS_IN_PROGRESS.dec()

    async def disconnect_all(self):
        for chat_session_id in self.active_connections:
            await self.disconnect(chat_session_id)

    async def receive(self, chat_session_id: int):
        connection: WebSocket = self.active_connections[chat_session_id]
        json_text = await connection.receive_text()
        payload = ChatRequest.parse_obj(json.loads(json_text))

        # TODO: Pre-tasks

        response = version_selector("v1")(
            payload.messages,
            system_message=payload.system_message,
            temperature=payload.temperature,
        )

        # TODO: Post-tasks

        await connection.send_text(response.model_dump_json())


def mark_chat_session_time_stopped(chat_session_id: int):
    with SessionLocal() as session:
        cs = session.query(ChatSession).filter_by(chat_session_id=chat_session_id).first()
        if cs:
            cs.time_stopped = datetime.datetime.now(datetime.timezone.utc)
            session.commit()
        else:
            logger.error(f"ChatSession.chat_session_id == {chat_session_id} not found")


def new_chat_session() -> int:
    with SessionLocal() as session:
        cs = ChatSession(time_started=datetime.datetime.now(datetime.timezone.utc))
        session.add(cs)
        session.commit()
        return cs.chat_session_id


def version_selector(version):
    if version == "v1":
        return v1_chat
    elif version == "v2":
        return v2_chat
    else:
        logger.warning("unknown version: %s, defaulting to v1", version)
        return v1_chat
