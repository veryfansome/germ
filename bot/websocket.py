from abc import ABC, abstractmethod
from fastapi import WebSocket
from prometheus_client import Gauge
from sqlalchemy.future import select as sql_select
from sqlalchemy.sql import desc
from starlette.websockets import WebSocketDisconnect, WebSocketState
import asyncio
import datetime
import logging

from bot.api.models import ChatMessage, ChatRequest, ChatResponse, ChatSessionSummary
from bot.chat import async_openai_client
from bot.db.models import AsyncSessionLocal, ChatSession, ChatRequestReceived, ChatResponseSent
from bot.graph.control_plane import ControlPlane
from settings.germ_settings import WEBSOCKET_CONNECTION_IDLE_TIMEOUT
from settings.openai_settings import MINI_MODEL, HTTPX_TIMEOUT

logger = logging.getLogger(__name__)

##
# Metrics

METRIC_CHAT_SESSIONS_IN_PROGRESS = Gauge(
    "chat_sessions_in_progress", "Number of connected sessions that have yet to be disconnected")


class WebSocketSendEventHandler(ABC):
    @abstractmethod
    async def on_send(self,
                      chat_response_sent_id: int,
                      chat_response: ChatResponse,
                      chat_session_id: int,
                      chat_request_received_id: int = None):
        pass


class WebSocketSender:
    def __init__(self, control_plane: ControlPlane, chat_session_id: int, connection: WebSocket,
                 send_event_handlers: list[WebSocketSendEventHandler] = None):
        self.chat_session_id = chat_session_id
        self.connection = connection
        self.control_plane = control_plane
        self.send_event_handlers: list[WebSocketSendEventHandler] = (
            send_event_handlers if send_event_handlers is not None else [])

    async def send_chat_response(self, chat_response: ChatResponse):
        await self.connection.send_text(chat_response.model_dump_json())
        chat_response_sent_id = await new_chat_response_sent(
            self.chat_session_id, chat_response, chat_request_received_id=None)

        _, time_occurred = await self.control_plane.add_chat_response(chat_response_sent_id)
        await self.control_plane.link_chat_response_to_chat_session(
            chat_response_sent_id, self.chat_session_id, time_occurred)

        for handler in self.send_event_handlers:
            _ = asyncio.create_task(
                handler.on_send(chat_response_sent_id, chat_response, self.chat_session_id))

    async def return_chat_response(self, chat_request_received_id: int, chat_response: ChatResponse):
        await self.connection.send_text(chat_response.model_dump_json())
        chat_response_sent_id = await new_chat_response_sent(
            self.chat_session_id, chat_response, chat_request_received_id=chat_request_received_id)

        _, time_occurred = await self.control_plane.add_chat_response(chat_response_sent_id)
        await self.control_plane.link_chat_response_to_chat_request(
            chat_request_received_id, chat_response_sent_id, self.chat_session_id, time_occurred)
        await self.control_plane.link_chat_response_to_chat_session(
            chat_response_sent_id, self.chat_session_id, time_occurred)

        for handler in self.send_event_handlers:
            _ = asyncio.create_task(
                handler.on_send(chat_response_sent_id, chat_response, self.chat_session_id,
                                chat_request_received_id=chat_request_received_id))


class SessionMonitor(ABC):
    @abstractmethod
    async def on_tick(self, chat_session_id: int, ws_sender: WebSocketSender):
        pass


class WebSocketReceiveEventHandler(ABC):
    @abstractmethod
    async def on_receive(self,
                         chat_session_id: int, chat_request_received_id: int,
                         chat_request: ChatRequest, ws_sender: WebSocketSender):
        pass


class WebSocketConnectionManager:
    def __init__(self, control_plane: ControlPlane):
        self.active_connections: dict[int, WebSocket] = {}
        self.monitor_tasks: dict[int, tuple[asyncio.Task, asyncio.Event]] = {}
        self.control_plane = control_plane
        self.receive_event_handlers: list[WebSocketReceiveEventHandler] = []
        self.send_event_handlers: list[WebSocketSendEventHandler] = []
        self.session_monitors: list[SessionMonitor] = []

    def add_send_event_handler(self, handler: WebSocketSendEventHandler):
        self.send_event_handlers.append(handler)

    def add_session_monitor(self, handler: SessionMonitor):
        self.session_monitors.append(handler)

    def add_receive_event_handler(self, handler: WebSocketReceiveEventHandler):
        self.receive_event_handlers.append(handler)

    async def conduct_chat_session(self, chat_session_id: int):
        try:
            while True:
                await asyncio.wait_for(self.receive(chat_session_id),
                                       timeout=WEBSOCKET_CONNECTION_IDLE_TIMEOUT)
        except asyncio.TimeoutError:
            logger.info(f"chat session {chat_session_id} timed out")
            await self.disconnect(chat_session_id)
        except WebSocketDisconnect:
            await self.disconnect(chat_session_id)

    async def connect(self, ws: WebSocket) -> int:
        chat_session_id = await new_chat_session()
        await ws.accept()
        self.active_connections[chat_session_id] = ws

        # Create and store a cancellation event, and start a monitor task
        cancel_event = asyncio.Event()
        monitor_task = asyncio.create_task(self.monitor_chat_session(chat_session_id, ws, cancel_event))
        self.monitor_tasks[chat_session_id] = (monitor_task, cancel_event)

        METRIC_CHAT_SESSIONS_IN_PROGRESS.inc()
        _ = asyncio.create_task(self.control_plane.add_chat_session(chat_session_id))
        return chat_session_id

    async def disconnect(self, chat_session_id: int):
        ws = self.active_connections.pop(chat_session_id, None)

        # Signal the monitor coroutine to exit
        if chat_session_id in self.monitor_tasks:
            monitor_task, cancel_event = self.monitor_tasks.pop(chat_session_id)
            cancel_event.set()   # Tell the monitor loop to exit
            await monitor_task

        if ws and ws.state in [WebSocketState.CONNECTED, WebSocketState.CONNECTING]:
            logger.info(f"disconnecting session {chat_session_id}, socket state: connected=%s connecting=%s",
                        ws.state == WebSocketState.CONNECTED, ws.state == WebSocketState.CONNECTING)
            await ws.close()

        METRIC_CHAT_SESSIONS_IN_PROGRESS.dec()
        await update_chat_session_time_stopped(chat_session_id)
        await update_chat_session_summary(chat_session_id)

    async def disconnect_all(self):
        await asyncio.gather(*[self.disconnect(chat_session_id) for chat_session_id in self.active_connections])

    async def monitor_chat_session(self, chat_session_id: int, ws: WebSocket, cancel_event: asyncio.Event):
        """
        Runs in the background to do periodic or event-driven checks.
        Closes automatically when cancel_event is set.
        """
        ws_sender = WebSocketSender(self.control_plane, chat_session_id, ws,
                                    send_event_handlers=self.send_event_handlers)
        try:
            while not cancel_event.is_set():
                # Wait for either 15s or the event to be set, whichever comes first.
                try:
                    await asyncio.wait_for(asyncio.shield(cancel_event.wait()), timeout=15.0)
                except asyncio.TimeoutError:
                    logger.info(f"chat session {chat_session_id} is still active")
                    for monitor in self.session_monitors:
                        asyncio.create_task(monitor.on_tick(chat_session_id, ws_sender))
        except asyncio.CancelledError:
            logger.info(f"Session monitor for {chat_session_id} was cancelled.")
        finally:
            # Clean-up logic if needed
            pass

    async def receive(self, chat_session_id: int):
        connection: WebSocket = self.active_connections[chat_session_id]
        chat_request = ChatRequest.model_validate(await connection.receive_json())

        chat_request_received_id = await new_chat_request_received(chat_session_id, chat_request)
        _, time_occurred = await self.control_plane.add_chat_request(chat_request_received_id)
        _ = asyncio.create_task(
            self.control_plane.link_chat_request_to_chat_session(
                chat_request_received_id, chat_session_id, time_occurred))

        ws_sender = WebSocketSender(self.control_plane, chat_session_id, connection,
                                    send_event_handlers=self.send_event_handlers)
        for handler in self.receive_event_handlers:
            _ = asyncio.create_task(
                handler.on_receive(chat_session_id, chat_request_received_id, chat_request, ws_sender))


async def get_chat_session_messages(chat_session_id: int) -> list[ChatMessage]:
    messages: list[ChatMessage] = []
    async with (AsyncSessionLocal() as rdb_session):
        async with rdb_session.begin():
            response_sent_select_stmt = sql_select(ChatResponseSent).where(
                ChatResponseSent.chat_session_id == chat_session_id
            ).order_by(
                desc(ChatResponseSent.chat_response_sent_id)
            )
            response_sent_select_result = await rdb_session.execute(response_sent_select_stmt)
            response_sent_record = response_sent_select_result.scalars().first()
            if response_sent_record:
                if response_sent_record.chat_request_received_id is not None:

                    request_received_select_stmt = sql_select(ChatRequestReceived).join(ChatResponseSent).filter(
                        ChatRequestReceived.chat_request_received_id == ChatResponseSent.chat_request_received_id,
                        ChatResponseSent.chat_response_sent_id == response_sent_record.chat_response_sent_id
                    )
                    request_received_select_result = await rdb_session.execute(request_received_select_stmt)
                    request_received_record = request_received_select_result.scalars().one_or_none()
                    if request_received_record:
                        for m in ChatRequest.model_validate(request_received_record.chat_request).messages:
                            messages.append(m)
                messages.append(
                    ChatResponse.model_validate(response_sent_record.chat_response))
    return messages


async def get_chat_session_summaries() -> list[ChatSessionSummary]:
    cs_list: list[ChatSessionSummary] = []
    async with (AsyncSessionLocal() as rdb_session):
        async with rdb_session.begin():
            chat_session_select_stmt = sql_select(ChatSession).where(
                ChatSession.is_hidden.is_(False),
                ChatSession.summary.is_not(None),
                ChatSession.time_stopped.is_not(None))
            chat_session_select_result = await rdb_session.execute(chat_session_select_stmt)
            chat_session_records = chat_session_select_result.scalars().all()
            for cs in chat_session_records:
                cs_list.append(ChatSessionSummary(
                    chat_session_id=cs.chat_session_id,
                    summary=cs.summary,
                    time_started=cs.time_started,
                    time_stopped=cs.time_stopped,
                ))
    return cs_list


async def new_chat_request_received(chat_session_id: int, chat_request: ChatRequest) -> int:
    async with (AsyncSessionLocal() as rdb_session):
        async with rdb_session.begin():
            rdb_record = ChatRequestReceived(
                chat_session_id=chat_session_id,
                chat_request=chat_request.model_dump(),
                time_received=datetime.datetime.now(datetime.timezone.utc),
            )
            rdb_session.add(rdb_record)
            await rdb_session.commit()
            return rdb_record.chat_request_received_id


async def new_chat_response_sent(chat_session_id: int, chat_response: ChatResponse,
                           chat_request_received_id: int = None) -> int:
    async with (AsyncSessionLocal() as rdb_session):
        async with rdb_session.begin():
            rdb_record = ChatResponseSent(
                chat_session_id=chat_session_id,
                chat_response=chat_response.model_dump(),
                time_sent=datetime.datetime.now(datetime.timezone.utc),
            ) if chat_request_received_id is None else ChatResponseSent(
                chat_session_id=chat_session_id,
                chat_request_received_id=chat_request_received_id,
                chat_response=chat_response.model_dump(),
                time_sent=datetime.datetime.now(datetime.timezone.utc),
            )
            rdb_session.add(rdb_record)
            await rdb_session.commit()
            return rdb_record.chat_response_sent_id


async def new_chat_session() -> int:
    async with (AsyncSessionLocal() as rdb_session):
        async with rdb_session.begin():
            rdb_record = ChatSession(time_started=datetime.datetime.now(datetime.timezone.utc))
            rdb_session.add(rdb_record)
            await rdb_session.commit()
            return rdb_record.chat_session_id


async def update_chat_session_is_hidden(chat_session_id: int):
    async with (AsyncSessionLocal() as rdb_session):
        async with rdb_session.begin():
            chat_session_select_stmt = sql_select(ChatSession).where(ChatSession.chat_session_id == chat_session_id)
            chat_session_select_result = await rdb_session.execute(chat_session_select_stmt)
            chat_session_record = chat_session_select_result.scalars().first()
            if chat_session_record:
                chat_session_record.is_hidden = True
                await rdb_session.commit()
            else:
                logger.error(f"ChatSession.chat_session_id == {chat_session_id} not found")


async def update_chat_session_summary(chat_session_id: int) -> int:
    messages = await get_chat_session_messages(chat_session_id)
    message_cnt = len(messages)
    if message_cnt > 1:  # A conversation should have at least a message and a reply
        completion = await async_openai_client.chat.completions.create(
            messages=([m.model_dump() for m in messages] + [{
                "role": "user", "content": " ".join((
                    "Summarize this conversation using no more than 20 words.",
                    "What did the I want?",
                    "What was your response?"
                ))
            }]),
            model=MINI_MODEL, n=1,
            timeout=HTTPX_TIMEOUT)
        async with (AsyncSessionLocal() as rdb_session):
            async with rdb_session.begin():
                chat_session_select_stmt = sql_select(ChatSession).where(ChatSession.chat_session_id == chat_session_id)
                chat_session_select_result = await rdb_session.execute(chat_session_select_stmt)
                chat_session_record = chat_session_select_result.scalars().first()
                if chat_session_record:
                    chat_session_record.summary = completion.choices[0].message.content
                    await rdb_session.commit()
                else:
                    logger.error(f"ChatSession.chat_session_id == {chat_session_id} not found")
    else:
        logger.info("skipped adding chat session summary")
    return message_cnt


async def update_chat_session_time_stopped(chat_session_id: int):
    async with (AsyncSessionLocal() as rdb_session):
        async with rdb_session.begin():
            chat_session_select_stmt = sql_select(ChatSession).where(ChatSession.chat_session_id == chat_session_id)
            chat_session_select_result = await rdb_session.execute(chat_session_select_stmt)
            chat_session_record = chat_session_select_result.scalars().first()
            if chat_session_record:
                chat_session_record.time_stopped = datetime.datetime.now(datetime.timezone.utc)
                await rdb_session.commit()
            else:
                logger.error(f"ChatSession.chat_session_id == {chat_session_id} not found")
