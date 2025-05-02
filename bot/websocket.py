from abc import ABC, abstractmethod
from datetime import datetime
from fastapi import WebSocket
from prometheus_client import Gauge
from sqlalchemy import Table, and_, insert, select
from sqlalchemy.ext.asyncio import async_sessionmaker as async_pg_session_maker
from starlette.websockets import WebSocketDisconnect, WebSocketState
from uuid import UUID, uuid5
import aiofiles
import asyncio
import logging

from bot.api.models import ChatRequest, ChatResponse
from bot.graph.control_plane import ControlPlane
from security.encryption import decrypt_integer, derive_key_from_passphrase, encrypt_integer
from settings.germ_settings import (DATA_DIR, ENCRYPTION_PASSWORD, UUID5_NS,
                                    WEBSOCKET_IDLE_TIMEOUT, WEBSOCKET_MONITOR_INTERVAL_SECONDS)

logger = logging.getLogger(__name__)

##
# Encryption

ENCRYPTION_KEY = derive_key_from_passphrase(ENCRYPTION_PASSWORD)

##
# Metrics

METRIC_CONVERSATIONS_IN_PROGRESS = Gauge(
    "conversations_in_progress", "Number of conversations that have yet to be disconnected")


class WebSocketDisconnectEventHandler(ABC):
    @abstractmethod
    async def on_disconnect(self, conversation_id: int):
        pass


class WebSocketSendEventHandler(ABC):
    @abstractmethod
    async def on_send(self, sent_message_id: int, chat_response: ChatResponse, conversation_id: int,
                      received_message_id: int = None):
        pass


class WebSocketSender:
    def __init__(self, chat_message_table: Table, control_plane: ControlPlane, conversation_id: int,
                 connection: WebSocket, pg_session_maker: async_pg_session_maker,
                 send_event_handlers: list[WebSocketSendEventHandler] = None):
        self.chat_message_table = chat_message_table
        self.connection = connection
        self.control_plane = control_plane
        self.conversation_id = conversation_id
        self.pg_session_maker = pg_session_maker
        self.send_event_handlers: list[WebSocketSendEventHandler] = (
            send_event_handlers if send_event_handlers is not None else [])

    async def send_message(self, chat_response: ChatResponse):
        chat_response.conversation_ident = encrypt_integer(self.conversation_id, ENCRYPTION_KEY)
        await self.connection.send_text(chat_response.model_dump_json())

        if chat_response.model != "none":
            async_tasks = []
            sent_message_id, dt_created = await self.new_chat_message_sent(self.conversation_id, chat_response)
            for handler in self.send_event_handlers:
                async_tasks.append(asyncio.create_task(
                    handler.on_send(sent_message_id, chat_response, self.conversation_id)
                ))
            await asyncio.gather(*async_tasks)

    async def send_reply(self, received_message_id: int, chat_response: ChatResponse):
        chat_response.conversation_ident = encrypt_integer(self.conversation_id, ENCRYPTION_KEY)
        await self.connection.send_text(chat_response.model_dump_json())

        if chat_response.model != "none":
            async_tasks = []
            sent_message_id, dt_created = await self.new_chat_message_sent(self.conversation_id, chat_response)
            await self.control_plane.add_chat_message(sent_message_id, dt_created)
            async_tasks.append(asyncio.create_task(
                self.control_plane.link_chat_message_sent_to_chat_message_received(
                    received_message_id, sent_message_id, self.conversation_id)
            ))
            for handler in self.send_event_handlers:
                async_tasks.append(asyncio.create_task(
                    handler.on_send(sent_message_id, chat_response, self.conversation_id,
                                    received_message_id=received_message_id)
                ))
            await asyncio.gather(*async_tasks)

    async def new_chat_message_sent(self, conversation_id: int, chat_response: ChatResponse) -> (int, datetime):
        # wrap chat_response in metadata related to time so we can repopulate using old conversations
        response_json = chat_response.model_dump_json()
        async with (self.pg_session_maker() as rdb_session):
            async with rdb_session.begin():
                try:
                    insert_stmt = insert(self.chat_message_table).values(
                        conversation_id=conversation_id,
                        json_sig=await persist_message_data(response_json),
                        received=False,
                    ).returning(
                        self.chat_message_table.c.message_id,
                        self.chat_message_table.c.dt_created
                    )
                    result = await rdb_session.execute(insert_stmt)
                    row = result.first()
                    await rdb_session.commit()
                    message_id, dt_created = row
                    logger.info(f"New message {message_id} inserted successfully "
                                f"for conversation {conversation_id} at {dt_created}")
                    return message_id, dt_created
                except Exception as e:
                    logger.error(f"Failed to insert new message with conversation {conversation_id}: {e}")
                    await rdb_session.rollback()
                    raise


class WebSocketReceiveEventHandler(ABC):
    @abstractmethod
    async def on_receive(self,
                         conversation_id: int, chat_request_received_id: int,
                         chat_request: ChatRequest, ws_sender: WebSocketSender):
        pass


class WebSocketSessionMonitor(ABC):
    @abstractmethod
    async def on_tick(self, conversation_id: int, ws_sender: WebSocketSender):
        pass


class WebSocketConnectionManager:
    def __init__(self, conversation_table: Table, chat_message_table: Table,
                 control_plane: ControlPlane, pg_session_maker: async_pg_session_maker):
        self.active_connections: dict[int, WebSocket] = {}
        self.chat_message_table = chat_message_table
        self.control_plane = control_plane
        self.conversation_table = conversation_table
        self.disconnect_event_handlers: list[WebSocketDisconnectEventHandler] = []
        self.monitor_tasks: dict[int, tuple[asyncio.Task, asyncio.Event]] = {}
        self.pg_session_maker = pg_session_maker
        self.receive_event_handlers: list[WebSocketReceiveEventHandler] = []
        self.send_event_handlers: list[WebSocketSendEventHandler] = []
        self.conversation_monitors: list[WebSocketSessionMonitor] = []

    def add_conversation_monitor(self, handler: WebSocketSessionMonitor):
        self.conversation_monitors.append(handler)

    def add_disconnect_event_handler(self, handler: WebSocketDisconnectEventHandler):
        self.disconnect_event_handlers.append(handler)

    def add_receive_event_handler(self, handler: WebSocketReceiveEventHandler):
        self.receive_event_handlers.append(handler)

    def add_send_event_handler(self, handler: WebSocketSendEventHandler):
        self.send_event_handlers.append(handler)

    async def connect(self, user_id: int, ws: WebSocket, conversation_ident: str = None) -> int:
        METRIC_CONVERSATIONS_IN_PROGRESS.inc()

        conversation_id = None if conversation_ident is None else decrypt_integer(conversation_ident, ENCRYPTION_KEY)
        if (conversation_id is None
                # Or stale/bad client-side data
                or await self.conversation_not_found(conversation_id, user_id)
                or conversation_id in self.active_connections):
            conversation_id, dt_created = await self.new_conversation(user_id)
            await self.control_plane.add_conversation(conversation_id, dt_created)
            await self.control_plane.link_chat_user_to_conversation(user_id, conversation_id)
        self.active_connections[conversation_id] = ws

        # Create and store a cancellation event, and start a monitor task
        cancel_event = asyncio.Event()
        monitor_task = asyncio.create_task(self.monitor_conversation(user_id, conversation_id, ws, cancel_event))
        self.monitor_tasks[conversation_id] = (monitor_task, cancel_event)
        return conversation_id

    async def conversation_not_found(self, conversation_id: int, user_id: int) -> bool:
        async with (self.pg_session_maker() as rdb_session):
            async with rdb_session.begin():
                chat_user_stmt = select(self.conversation_table.c.dt_modified).where(
                    and_(
                        self.conversation_table.c.conversation_id == conversation_id,
                        self.conversation_table.c.user_id == user_id,
                    )
                )
                record = (await rdb_session.execute(chat_user_stmt)).one_or_none()
                return record is None

    async def disconnect(self, conversation_id: int):
        ws = self.active_connections.pop(conversation_id, None)

        # Signal the monitor coroutine to exit
        if conversation_id in self.monitor_tasks:
            monitor_task, cancel_event = self.monitor_tasks.pop(conversation_id)
            cancel_event.set()   # Tell the monitor loop to exit
            await monitor_task

        logger.info(f"disconnecting conversation {conversation_id} {ws.client_state}")
        if ws.client_state == WebSocketState.CONNECTED:
            await ws.close()

        async_tasks = []
        for handler in self.disconnect_event_handlers:
            async_tasks.append(asyncio.create_task(handler.on_disconnect(conversation_id)))
        await asyncio.gather(*async_tasks)

        METRIC_CONVERSATIONS_IN_PROGRESS.dec()

    async def disconnect_all(self):
        await asyncio.gather(*[self.disconnect(conversation_id) for conversation_id in self.active_connections])

    async def monitor_conversation(self, user_id, conversation_id: int, ws: WebSocket, cancel_event: asyncio.Event):
        """
        Runs in the background to do periodic or event-driven checks.
        Closes automatically when cancel_event is set.
        """
        ws_sender = self.new_ws_sender(conversation_id, ws)
        try:
            while not cancel_event.is_set():
                # Wait for some seconds or until the cancel_event is set, whichever comes first.
                try:
                    await asyncio.wait_for(asyncio.shield(cancel_event.wait()),
                                           timeout=WEBSOCKET_MONITOR_INTERVAL_SECONDS)
                except asyncio.TimeoutError:
                    for monitor in self.conversation_monitors:
                        _ = asyncio.create_task(monitor.on_tick(conversation_id, ws_sender))
        except asyncio.CancelledError:
            logger.info(f"Session monitor for {conversation_id} was cancelled.")
        finally:
            # Clean-up logic if needed
            pass

    async def new_chat_message_received(self, conversation_id: int, chat_request: ChatRequest) -> (int, datetime):
        request_json = chat_request.model_dump_json()
        async with (self.pg_session_maker() as rdb_session):
            async with rdb_session.begin():
                try:
                    insert_stmt = insert(self.chat_message_table).values(
                        conversation_id=conversation_id,
                        json_sig=await persist_message_data(request_json),
                        received=True,
                    ).returning(
                        self.chat_message_table.c.message_id,
                        self.chat_message_table.c.dt_created
                    )
                    result = await rdb_session.execute(insert_stmt)
                    row = result.first()
                    await rdb_session.commit()
                    message_id, dt_created = row
                    logger.info(f"New message {message_id} inserted successfully "
                                f"for conversation {conversation_id} at {dt_created}")
                    return message_id, dt_created
                except Exception as e:
                    logger.error(f"Failed to insert new message with conversation {conversation_id}: {e}")
                    await rdb_session.rollback()
                    raise

    async def new_conversation(self, user_id: int) -> (int, datetime):
        async with (self.pg_session_maker() as rdb_session):
            async with rdb_session.begin():
                try:
                    insert_stmt = insert(self.conversation_table).values(
                        user_id=user_id,
                    ).returning(
                        self.conversation_table.c.conversation_id,
                        self.conversation_table.c.dt_created
                    )
                    result = await rdb_session.execute(insert_stmt)
                    row = result.first()
                    await rdb_session.commit()
                    conversation_id, dt_created = row
                    logger.info(f"New conversation {conversation_id} with user {user_id} "
                                f"inserted successfully at {dt_created}")
                    return conversation_id, dt_created
                except Exception as e:
                    logger.error(f"Failed to insert new conversation with user {user_id}: {e}")
                    await rdb_session.rollback()
                    raise

    def new_ws_sender(self, conversation_id: int, ws: WebSocket):
        return WebSocketSender(self.chat_message_table, self.control_plane, conversation_id, ws, self.pg_session_maker,
                               send_event_handlers=self.send_event_handlers)

    async def receive(self, user_id: int, conversation_id: int):
        ws: WebSocket = self.active_connections[conversation_id]
        chat_request = ChatRequest.model_validate(await ws.receive_json())

        async_tasks = []
        message_id, dt_created = await self.new_chat_message_received(conversation_id, chat_request)
        await self.control_plane.add_chat_message(message_id, dt_created)
        async_tasks.append(asyncio.create_task(
            self.control_plane.link_chat_message_received_to_chat_user(message_id, user_id)
        ))
        ws_sender = self.new_ws_sender(conversation_id, ws)
        for handler in self.receive_event_handlers:
            async_tasks.append(asyncio.create_task(
                handler.on_receive(conversation_id, message_id, chat_request, ws_sender)
            ))
        await asyncio.gather(*async_tasks)

    async def wait_for_receive(self, user_id: int, conversation_id: int):
        try:
            while True:
                await asyncio.wait_for(self.receive(user_id, conversation_id), timeout=WEBSOCKET_IDLE_TIMEOUT)
        except asyncio.TimeoutError:
            logger.info(f"Conversation {conversation_id} with user {user_id} timed out")
            await self.disconnect(conversation_id)
        except WebSocketDisconnect:
            await self.disconnect(conversation_id)


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
