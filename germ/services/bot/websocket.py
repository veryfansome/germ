import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from fastapi import WebSocket
from opentelemetry import trace
from sqlalchemy import Table, and_, insert, select
from sqlalchemy.ext.asyncio import async_sessionmaker as async_pg_session_maker
from starlette.websockets import WebSocketDisconnect
from traceback import format_exc

from germ.api.models import ChatRequest, ChatResponse
from germ.database.neo4j import KnowledgeGraph
from germ.security.encryption import decrypt_integer, derive_key_from_passphrase, encrypt_integer
from germ.settings.germ_settings import (GERM_ENCRYPTION_PASSWORD, GERM_WEBSOCKET_IDLE_TIMEOUT,
                                         GERM_WEBSOCKET_MONITOR_INTERVAL_SECONDS)

logger = logging.getLogger(__name__)

tracer = trace.get_tracer(__name__)

ENCRYPTION_KEY = derive_key_from_passphrase(GERM_ENCRYPTION_PASSWORD)


class WebSocketDisconnectEventHandler(ABC):
    @abstractmethod
    async def on_disconnect(self, conversation_id: int):
        pass


class WebSocketSendEventHandler(ABC):
    @abstractmethod
    async def on_send(self, conversation_id: int, dt_created: datetime,
                      chat_response: ChatResponse, received_message_dt_created: datetime = None):
        pass


class WebSocketSender(ABC):
    @abstractmethod
    async def send_message(self, chat_response: ChatResponse):
        pass

    @abstractmethod
    async def send_reply(self, received_message_dt_created: datetime, chat_response: ChatResponse):
        pass


class WebSocketReceiveEventHandler(ABC):
    @abstractmethod
    async def on_receive(self, session, conversation_id: int, dt_created: datetime,
                         chat_request: ChatRequest, ws_sender: WebSocketSender):
        pass


class WebSocketSessionMonitor(ABC):
    @abstractmethod
    async def on_tick(self, conversation_id: int, ws_sender: WebSocketSender):
        pass


class DefaultWebSocketSender(WebSocketSender):
    def __init__(self, knowledge_graph: KnowledgeGraph, conversation_id: int,
                 connection: WebSocket, send_event_handlers: list[WebSocketSendEventHandler] = None):
        self.connection = connection
        self.knowledge_graph = knowledge_graph
        self.conversation_id = conversation_id
        self.send_event_handlers: list[WebSocketSendEventHandler] = (
            send_event_handlers if send_event_handlers is not None else []
        )

    async def send_message(self, chat_response: ChatResponse):
        chat_response.conversation_ident = encrypt_integer(self.conversation_id, ENCRYPTION_KEY)
        await self.connection.send_text(chat_response.model_dump_json())
        if chat_response.model != "none":
            async_tasks = []
            dt_created = datetime.now(tz=timezone.utc)
            for handler in self.send_event_handlers:
                async_tasks.append(asyncio.create_task(
                    handler.on_send(self.conversation_id, dt_created, chat_response)
                ))
            await asyncio.gather(*async_tasks)

    async def send_reply(self, received_message_dt_created: datetime, chat_response: ChatResponse):
        chat_response.conversation_ident = encrypt_integer(self.conversation_id, ENCRYPTION_KEY)
        await self.connection.send_text(chat_response.model_dump_json())
        if chat_response.model != "none":
            dt_created = datetime.now(tz=timezone.utc)
            await self.knowledge_graph.add_chat_message(self.conversation_id, dt_created)
            async_tasks = [
                asyncio.create_task(
                    self.knowledge_graph.link_chat_message_sent_to_chat_message_received(
                        received_message_dt_created, dt_created, self.conversation_id)
                ),
                asyncio.create_task(
                    self.knowledge_graph.link_chat_message_to_conversation(self.conversation_id, dt_created)
                )
            ]
            for handler in self.send_event_handlers:
                async_tasks.append(asyncio.create_task(
                    handler.on_send(self.conversation_id, dt_created, chat_response,
                                    received_message_dt_created=received_message_dt_created)
                ))
            await asyncio.gather(*async_tasks)


class WebSocketConnectionManager:
    def __init__(self, conversation_table: Table, conversation_state: Table, struct_type_table: Table,
                 knowledge_graph: KnowledgeGraph, pg_session_maker: async_pg_session_maker):
        self.active_connections: dict[int, WebSocket] = {}
        self.knowledge_graph: KnowledgeGraph = knowledge_graph
        self.conversation_monitors: list[WebSocketSessionMonitor] = []
        self.conversation_state: Table = conversation_state
        self.conversation_table: Table = conversation_table
        self.conversations: dict[int, dict] = {}
        self.disconnect_event_handlers: list[WebSocketDisconnectEventHandler] = []
        self.monitor_tasks: dict[int, tuple[asyncio.Task, asyncio.Event]] = {}
        self.pg_session_maker: async_pg_session_maker = pg_session_maker
        self.receive_event_handlers: list[WebSocketReceiveEventHandler] = []
        self.send_event_handlers: list[WebSocketSendEventHandler] = []

    def add_conversation_monitor(self, handler: WebSocketSessionMonitor):
        self.conversation_monitors.append(handler)

    def add_disconnect_event_handler(self, handler: WebSocketDisconnectEventHandler):
        self.disconnect_event_handlers.append(handler)

    def add_receive_event_handler(self, handler: WebSocketReceiveEventHandler):
        self.receive_event_handlers.append(handler)

    def add_send_event_handler(self, handler: WebSocketSendEventHandler):
        self.send_event_handlers.append(handler)

    async def connect(self, session, ws: WebSocket, conversation_ident: str = None) -> int | None:
        try:
            conversation_id = None if conversation_ident is None else decrypt_integer(conversation_ident, ENCRYPTION_KEY)
        except Exception:
            return None

        if (conversation_id is None
                # Or stale/bad client-side data
                or await self.conversation_not_found(conversation_id, session["user_id"])
                or conversation_id in self.active_connections):
            conversation_id, dt_created = await self.new_conversation(session["user_id"])
            await self.knowledge_graph.add_conversation(conversation_id, dt_created)
            asyncio.create_task(
                self.knowledge_graph.link_chat_user_to_conversation(session["user_id"], conversation_id)
            )
        self.active_connections[conversation_id] = ws

        # Create and store a cancellation event, and start a monitor task
        cancel_event = asyncio.Event()
        monitor_task = asyncio.create_task(self.monitor_conversation(session, conversation_id, ws, cancel_event))
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

    async def disconnect(self, conversation_id: int, close_ws: bool = True):
        ws = self.active_connections.pop(conversation_id, None)

        # Signal the monitor coroutine to exit
        if conversation_id in self.monitor_tasks:
            monitor_task, cancel_event = self.monitor_tasks.pop(conversation_id)
            cancel_event.set()   # Tell the monitor loop to exit
            await monitor_task

        if close_ws:
            logger.info(f"Disconnecting conversation {conversation_id}")
            await ws.close()

        async_tasks = [asyncio.create_task(handler.on_disconnect(conversation_id))
                       for handler in self.disconnect_event_handlers]
        await asyncio.gather(*async_tasks)

    async def disconnect_all(self):
        await asyncio.gather(*[self.disconnect(conversation_id) for conversation_id in self.active_connections])

    async def monitor_conversation(self, session, conversation_id: int, ws: WebSocket, cancel_event: asyncio.Event):
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
                                           timeout=GERM_WEBSOCKET_MONITOR_INTERVAL_SECONDS)
                except asyncio.TimeoutError:
                    # TODO: Update conversation_state
                    async_tasks = [asyncio.create_task(monitor.on_tick(conversation_id, ws_sender))
                                   for monitor in self.conversation_monitors]
                    await asyncio.gather(*async_tasks)
        except asyncio.CancelledError:
            logger.info(f"Session monitor for {conversation_id} was cancelled.")

    async def new_conversation(self, user_id: int) -> (int, datetime):
        async with (self.pg_session_maker() as rdb_session):
            async with rdb_session.begin():
                try:
                    # Insert a new conversation record
                    conversation_insert_stmt = insert(self.conversation_table).values(
                        user_id=user_id,
                    ).returning(
                        self.conversation_table.c.conversation_id,
                        self.conversation_table.c.dt_created
                    )
                    conversation_result = await rdb_session.execute(conversation_insert_stmt)
                    conversation_row = conversation_result.first()
                    conversation_id, dt_created = conversation_row
                    # Insert a new conversation state record
                    conversation_state_insert_stmt = insert(self.conversation_state).values(
                        conversation_id=conversation_id,
                        status=1,
                    )
                    await rdb_session.execute(conversation_state_insert_stmt)
                    await rdb_session.commit()
                    logger.info(f"New conversation {conversation_id} with user {user_id} "
                                f"inserted successfully at {dt_created}.")
                    return conversation_id, dt_created
                except Exception as e:
                    logger.error(f"Failed to insert new conversation with user {user_id}.\n{format_exc()}")
                    await rdb_session.rollback()
                    raise

    def new_ws_sender(self, conversation_id: int, ws: WebSocket) -> WebSocketSender:
        return DefaultWebSocketSender(self.knowledge_graph, conversation_id, ws,
                                      send_event_handlers=self.send_event_handlers)

    async def receive(self, ws: WebSocket, ws_sender: WebSocketSender, session, conversation_id: int):
        try:
            chat_request = ChatRequest.model_validate(await ws.receive_json())

            dt_created = datetime.now(tz=timezone.utc)
            await self.knowledge_graph.add_chat_message(conversation_id, dt_created)

            async_tasks = ([
                asyncio.create_task(
                    self.knowledge_graph.link_chat_message_received_to_chat_user(
                        conversation_id, dt_created, session["user_id"]
                    )
                ),
                asyncio.create_task(
                    self.knowledge_graph.link_chat_message_to_conversation(conversation_id, dt_created)
                )
            ])
            async_tasks.extend(
                asyncio.create_task(
                    handler.on_receive(session, conversation_id, dt_created, chat_request, ws_sender)
                )
                for handler in self.receive_event_handlers
            )
            await asyncio.gather(*async_tasks)
        except WebSocketDisconnect:
            raise  # Pass through WebSocketDisconnect to the caller
        except Exception:
            logger.error(f"Error in conversation {conversation_id} with user {session['user_id']}.\n{format_exc()}")
            await ws_sender.send_message(
                ChatResponse(complete=True, content="An error occurred while processing your request.", error=True)
            )

    async def wait_for_receive(self, session, conversation_id: int):
        ws: WebSocket = self.active_connections[conversation_id]
        ws_sender = self.new_ws_sender(conversation_id, ws)
        try:
            await ws_sender.send_message(
                ChatResponse(complete=True, content="Connected!")
            )
            while True:
                await asyncio.wait_for(
                    self.receive(ws, ws_sender, session, conversation_id),
                    timeout=GERM_WEBSOCKET_IDLE_TIMEOUT
                )
        except asyncio.TimeoutError:
            logger.info(f"Conversation {conversation_id} with user {session['user_id']} timed out.")
            await self.disconnect(conversation_id)
        except WebSocketDisconnect:
            logger.info(f"Conversation {conversation_id} with user {session['user_id']} disconnected.")
            await self.disconnect(conversation_id, close_ws=False)  # Skip close since already disconnected
