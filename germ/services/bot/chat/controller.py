import json
import logging
import time
from datetime import datetime
from pydantic import BaseModel

from germ.api.models import ChatRequest, ChatResponse
from germ.observability.annotations import measure_exec_seconds
from germ.services.bot.chat.classifier import ChatMessageMetadata, ChatRequestClassifier
from germ.services.bot.websocket import (WebSocketDisconnectEventHandler, WebSocketReceiveEventHandler,
                                         WebSocketSendEventHandler, WebSocketSender, WebSocketSessionMonitor)

logger = logging.getLogger(__name__)
message_logger = logging.getLogger('message')


class ConversationMetadata(BaseModel):
    conversation_id: int
    messages: dict[datetime, ChatMessageMetadata] = {}

    async def add_message(self, dt_created: datetime, message_meta: ChatMessageMetadata, log: bool = False):
        self.messages[dt_created] = message_meta
        if log:
            message_logger.info(
                json.dumps(
                    [int(dt_created.timestamp()), self.conversation_id, message_meta.model_dump(exclude_none=True)],
                    separators=(",", ":")
                )
            )


class ChatController(WebSocketDisconnectEventHandler, WebSocketReceiveEventHandler,
                     WebSocketSendEventHandler, WebSocketSessionMonitor):
    def __init__(
            self, request_classifier: ChatRequestClassifier,
            delegate: WebSocketReceiveEventHandler,
    ):
        self.delegate = delegate
        self.conversations: dict[int, ConversationMetadata] = {}
        self.request_classifier = request_classifier

    async def on_disconnect(self, conversation_id: int):
        pass

    @measure_exec_seconds(use_logging=True, use_prometheus=True)
    async def on_receive(self, user_id: int, conversation_id: int, dt_created: datetime,
                         chat_request: ChatRequest, ws_sender: WebSocketSender):
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = ConversationMetadata(conversation_id=conversation_id)

        message_meta = await self.request_classifier.classify_request(user_id, chat_request)
        await self.conversations[conversation_id].add_message(dt_created, message_meta, log=True)

        # Send to LLM
        await self.delegate.on_receive(user_id, conversation_id, dt_created, chat_request, ws_sender)

    async def on_send(self, conversation_id: int, dt_created: datetime,
                      chat_response: ChatResponse, received_message_dt_created: datetime = None):
        if conversation_id not in self.conversations:
            logger.error(f"Conversation {conversation_id} not found.")
            return

        message_meta = await self.request_classifier.classify_response(chat_response)
        await self.conversations[conversation_id].add_message(dt_created, message_meta, log=True)

    async def on_start(self):
        pass

    async def on_tick(self, conversation_id: int, ws_sender: WebSocketSender):
        if conversation_id not in self.conversations:
            logger.error(f"Conversation {conversation_id} not found.")
            return

        timestamps = list(self.conversations[conversation_id].messages.keys())
        timestamps.sort()
        last_message_age_secs = int(time.time()) - timestamps.pop().timestamp()
        logger.info(f"{last_message_age_secs} seconds since last message on conversation {conversation_id}")
