from abc import ABC, abstractmethod
from datetime import datetime
from fastapi import WebSocket
from prometheus_client import Gauge
from sqlalchemy import Table, insert
from sqlalchemy.ext.asyncio import async_sessionmaker as async_pg_session_maker
from starlette.websockets import WebSocketDisconnect, WebSocketState
from uuid import UUID, uuid5
import aiofiles
import asyncio
import logging

from bot.api.models import ChatRequest, ChatResponse
from bot.graph.control_plane import ControlPlane
from settings.germ_settings import DATA_DIR, UUID5_NS, WEBSOCKET_IDLE_TIMEOUT, WEBSOCKET_MONITOR_INTERVAL_SECONDS

logger = logging.getLogger(__name__)

##
# Metrics

METRIC_CHAT_SESSIONS_IN_PROGRESS = Gauge(
    "chat_sessions_in_progress", "Number of connected sessions that have yet to be disconnected")


class WebSocketDisconnectEventHandler(ABC):
    @abstractmethod
    async def on_disconnect(self, chat_session_id: int):
        pass


class WebSocketSendEventHandler(ABC):
    @abstractmethod
    async def on_send(self, sent_message_id: int, chat_response: ChatResponse, chat_session_id: int,
                      received_message_id: int = None):
        pass


class WebSocketSender:
    def __init__(self, chat_message_table: Table, control_plane: ControlPlane, chat_session_id: int,
                 connection: WebSocket, pg_session_maker: async_pg_session_maker,
                 send_event_handlers: list[WebSocketSendEventHandler] = None):
        self.chat_message_table = chat_message_table
        self.chat_session_id = chat_session_id
        self.connection = connection
        self.control_plane = control_plane
        self.pg_session_maker = pg_session_maker
        self.send_event_handlers: list[WebSocketSendEventHandler] = (
            send_event_handlers if send_event_handlers is not None else [])

    async def send_message(self, chat_response: ChatResponse):
        await self.connection.send_text(chat_response.model_dump_json())

        async_tasks = []
        sent_message_id, dt_created = await self.new_chat_message_sent(self.chat_session_id, chat_response)
        for handler in self.send_event_handlers:
            async_tasks.append(asyncio.create_task(
                handler.on_send(sent_message_id, chat_response, self.chat_session_id)
            ))
        await asyncio.gather(*async_tasks)

    async def send_reply(self, received_message_id: int, chat_response: ChatResponse):
        await self.connection.send_text(chat_response.model_dump_json())

        async_tasks = []
        sent_message_id, dt_created = await self.new_chat_message_sent(self.chat_session_id, chat_response)
        await self.control_plane.add_chat_message(sent_message_id, dt_created)
        async_tasks.append(asyncio.create_task(
            self.control_plane.link_chat_message_sent_to_chat_message_received(
                received_message_id, sent_message_id, self.chat_session_id)
        ))
        for handler in self.send_event_handlers:
            async_tasks.append(asyncio.create_task(
                handler.on_send(sent_message_id, chat_response, self.chat_session_id,
                                received_message_id=received_message_id)
            ))
        await asyncio.gather(*async_tasks)

    async def new_chat_message_sent(self, session_id: int, chat_response: ChatResponse) -> (int, datetime):
        response_json = chat_response.model_dump_json()
        async with (self.pg_session_maker() as rdb_session):
            async with rdb_session.begin():
                try:
                    insert_stmt = insert(self.chat_message_table).values(
                        json_sig=await persist_message_data(response_json),
                        received=False,
                        session_id=session_id,
                    ).returning(
                        self.chat_message_table.c.message_id,
                        self.chat_message_table.c.dt_created
                    )
                    result = await rdb_session.execute(insert_stmt)
                    row = result.first()
                    await rdb_session.commit()
                    message_id, dt_created = row
                    logger.info(f"New message '{message_id}' inserted successfully "
                                f"for session {session_id} at {dt_created}")
                    return message_id, dt_created
                except Exception as e:
                    logger.error(f"Failed to insert new message with session {session_id}: {e}")
                    await rdb_session.rollback()
                    raise


class WebSocketSessionMonitor(ABC):
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
    def __init__(self, chat_message_table: Table, chat_session_table: Table,
                 control_plane: ControlPlane, pg_session_maker: async_pg_session_maker):
        self.active_connections: dict[int, WebSocket] = {}
        self.chat_message_table = chat_message_table
        self.chat_session_table = chat_session_table
        self.control_plane = control_plane
        self.disconnect_event_handlers: list[WebSocketDisconnectEventHandler] = []
        self.monitor_tasks: dict[int, tuple[asyncio.Task, asyncio.Event]] = {}
        self.pg_session_maker = pg_session_maker
        self.receive_event_handlers: list[WebSocketReceiveEventHandler] = []
        self.send_event_handlers: list[WebSocketSendEventHandler] = []
        self.session_monitors: list[WebSocketSessionMonitor] = []

    def add_disconnect_event_handler(self, handler: WebSocketDisconnectEventHandler):
        self.disconnect_event_handlers.append(handler)

    def add_receive_event_handler(self, handler: WebSocketReceiveEventHandler):
        self.receive_event_handlers.append(handler)

    def add_send_event_handler(self, handler: WebSocketSendEventHandler):
        self.send_event_handlers.append(handler)

    def add_session_monitor(self, handler: WebSocketSessionMonitor):
        self.session_monitors.append(handler)

    async def connect(self, user_id: int, ws: WebSocket) -> int:
        session_id, dt_created = await self.new_chat_session(user_id)
        self.active_connections[session_id] = ws

        # Create and store a cancellation event, and start a monitor task
        cancel_event = asyncio.Event()
        monitor_task = asyncio.create_task(self.monitor_chat_session(user_id, session_id, ws, cancel_event))
        self.monitor_tasks[session_id] = (monitor_task, cancel_event)

        METRIC_CHAT_SESSIONS_IN_PROGRESS.inc()
        await self.control_plane.add_chat_session(session_id, dt_created)
        await self.control_plane.link_chat_user_to_chat_session(user_id, session_id)
        return session_id

    async def disconnect(self, chat_session_id: int):
        ws = self.active_connections.pop(chat_session_id, None)

        # Signal the monitor coroutine to exit
        if chat_session_id in self.monitor_tasks:
            monitor_task, cancel_event = self.monitor_tasks.pop(chat_session_id)
            cancel_event.set()   # Tell the monitor loop to exit
            await monitor_task

        logger.info(f"disconnecting session {chat_session_id} {ws.client_state}")
        if ws.client_state == WebSocketState.CONNECTED:
            await ws.close()

        METRIC_CHAT_SESSIONS_IN_PROGRESS.dec()
        async_tasks = []
        for handler in self.disconnect_event_handlers:
            async_tasks.append(asyncio.create_task(handler.on_disconnect(chat_session_id)))
        await asyncio.gather(*async_tasks)

    async def disconnect_all(self):
        await asyncio.gather(*[self.disconnect(chat_session_id) for chat_session_id in self.active_connections])

    async def monitor_chat_session(self, user_id, chat_session_id: int, ws: WebSocket, cancel_event: asyncio.Event):
        """
        Runs in the background to do periodic or event-driven checks.
        Closes automatically when cancel_event is set.
        """
        ws_sender = WebSocketSender(self.chat_message_table, self.control_plane, chat_session_id, ws, self.pg_session_maker,
                                    send_event_handlers=self.send_event_handlers)
        try:
            while not cancel_event.is_set():
                # Wait for some seconds or until the cancel_event is set, whichever comes first.
                try:
                    await asyncio.wait_for(asyncio.shield(cancel_event.wait()),
                                           timeout=WEBSOCKET_MONITOR_INTERVAL_SECONDS)
                except asyncio.TimeoutError:
                    for monitor in self.session_monitors:
                        _ = asyncio.create_task(monitor.on_tick(chat_session_id, ws_sender))
        except asyncio.CancelledError:
            logger.info(f"Session monitor for {chat_session_id} was cancelled.")
        finally:
            # Clean-up logic if needed
            pass

    async def new_chat_message_received(self, session_id: int, chat_request: ChatRequest) -> (int, datetime):
        request_json = chat_request.model_dump_json()
        async with (self.pg_session_maker() as rdb_session):
            async with rdb_session.begin():
                try:
                    insert_stmt = insert(self.chat_message_table).values(
                        json_sig=await persist_message_data(request_json),
                        received=True,
                        session_id=session_id,
                    ).returning(
                        self.chat_message_table.c.message_id,
                        self.chat_message_table.c.dt_created
                    )
                    result = await rdb_session.execute(insert_stmt)
                    row = result.first()
                    await rdb_session.commit()
                    message_id, dt_created = row
                    logger.info(f"New message '{message_id}' inserted successfully "
                                f"for session {session_id} at {dt_created}")
                    return message_id, dt_created
                except Exception as e:
                    logger.error(f"Failed to insert new message with session {session_id}: {e}")
                    await rdb_session.rollback()
                    raise

    async def new_chat_session(self, user_id: int) -> (int, datetime):
        async with (self.pg_session_maker() as rdb_session):
            async with rdb_session.begin():
                try:
                    insert_stmt = insert(self.chat_session_table).values(
                        user_id=user_id,
                    ).returning(
                        self.chat_session_table.c.session_id,
                        self.chat_session_table.c.dt_created
                    )
                    result = await rdb_session.execute(insert_stmt)
                    row = result.first()
                    await rdb_session.commit()
                    session_id, dt_created = row
                    logger.info(f"New session '{session_id}' with user {user_id} "
                                f"inserted successfully at {dt_created}")
                    return session_id, dt_created
                except Exception as e:
                    logger.error(f"Failed to insert new session with user {user_id}: {e}")
                    await rdb_session.rollback()
                    raise

    async def receive(self, user_id: int, session_id: int):
        ws: WebSocket = self.active_connections[session_id]
        chat_request = ChatRequest.model_validate(await ws.receive_json())

        async_tasks = []
        message_id, dt_created = await self.new_chat_message_received(session_id, chat_request)
        await self.control_plane.add_chat_message(message_id, dt_created)
        async_tasks.append(asyncio.create_task(
            self.control_plane.link_chat_message_received_to_chat_user(message_id, user_id)
        ))
        ws_sender = WebSocketSender(self.chat_message_table, self.control_plane, session_id, ws, self.pg_session_maker,
                                    send_event_handlers=self.send_event_handlers)
        for handler in self.receive_event_handlers:
            async_tasks.append(asyncio.create_task(
                handler.on_receive(session_id, message_id, chat_request, ws_sender)
            ))
        await asyncio.gather(*async_tasks)

    async def wait_for_receive(self, user_id: int, chat_session_id: int):
        try:
            while True:
                await asyncio.wait_for(self.receive(user_id, chat_session_id), timeout=WEBSOCKET_IDLE_TIMEOUT)
        except asyncio.TimeoutError:
            logger.info(f"chat session {chat_session_id} with user {user_id} timed out")
            await self.disconnect(chat_session_id)
        except WebSocketDisconnect:
            await self.disconnect(chat_session_id)


async def persist_message_data(json_blob) -> UUID:
    json_sig = uuid5(UUID5_NS, json_blob)
    data_filename = f"{DATA_DIR}/messages/{json_sig}.json"
    try:
        async with aiofiles.open(data_filename, "w") as f:
            await f.write(json_blob)
        return json_sig
    except Exception as e:
        logger.error(f"Failed to write {data_filename}: {e}")
        raise
