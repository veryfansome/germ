from typing_extensions import Literal
import json
import tiktoken

from api.models import ChatMessage, ChatResponse
from bot.openai_utils import (CHAT_COMPLETION_FUNCTIONS,
                              DEFAULT_CHAT_MODEL,
                              ENABLED_TOOLS, handle_tool_calls_in_completion_response, is_token_limit_check_enabled)
from db.models import MessageReceived, MessageReplied, SessionLocal
from observability.logging import logging

logger = logging.getLogger(__name__)


# This function provides:
# - Basic chat completion with limited tools
# - Chat history
def chat(messages: list[ChatMessage],
         model: str = DEFAULT_CHAT_MODEL,
         system_message: str = None,
         temperature: float = 0.0,
         tools: dict[str, dict] = ENABLED_TOOLS,
         tool_choice: Literal['auto', 'none'] = 'auto') -> ChatResponse:
    new_chat_message: ChatMessage = messages[-1]
    logger.debug("received: %s", new_chat_message.content)

    # A `chat_frame` is a list of `ChatMessage`s. It should be a list with a single element for new chats and a series
    # of messages between the `user` and `assistant`.
    chat_frame = messages
    if is_token_limit_check_enabled(model):
        # Trim message list to avoid hitting selected model's token limit.
        enc = tiktoken.encoding_for_model(model)
        total_tokens = len(enc.encode(system_message))
        reversed_chat_frame = []
        for chat_message in reversed(messages):  # In reverse because message list is ordered from oldest to newest.
            message_tokens = len(enc.encode(chat_message.content))
            # If adding `message_tokens` pushes us over the limit, stop appending
            if message_tokens + total_tokens > enc.max_token_value:
                break
            total_tokens += message_tokens
            reversed_chat_frame.append(chat_message)
        chat_frame = tuple(reversed(reversed_chat_frame))  # Undo previously reversed order

    # Update message history
    message_received = MessageReceived(
        chat_frame=[f.dict() for f in chat_frame[:-1]],
        content=new_chat_message.content.encode('utf-8'),
        role=new_chat_message.role,
    )
    with SessionLocal() as session:
        session.add(message_received)
        session.commit()
        session.refresh(message_received)

    # Do completion request
    completion = CHAT_COMPLETION_FUNCTIONS[model](chat_frame,
                                                  system_message=system_message,
                                                  temperature=temperature,
                                                  tools=tools,
                                                  tool_choice=tool_choice)
    completion_message = handle_tool_calls_in_completion_response(completion)
    completion.choices[0].message = completion_message

    # Update message history
    message_replied = MessageReplied(
        content=None if completion_message.tool_calls else completion_message.content.encode('utf-8'),
        message_received_id=message_received.id,
        role=completion_message.role,
        tool_calls=None if not completion_message.tool_calls else [c.dict() for c in completion_message.tool_calls]
    )
    with SessionLocal() as session:
        session.add(message_replied)
        session.commit()
        session.refresh(message_replied)
    return ChatResponse(
        message_received_id=message_received.id,
        message_replied_id=message_replied.id,
        response=completion,
    )
