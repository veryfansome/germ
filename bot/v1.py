from typing_extensions import Literal
import json
import tiktoken

from api.models import ChatMessage
from bot.openai_utils import (CHAT_COMPLETION_FUNCTIONS,
                              DEFAULT_CHAT_MODEL,
                              ENABLED_TOOLS, is_token_limit_check_enabled, tool_wrapper)
from db.models import MessageReceived, MessageReplied, SessionLocal
from observability.logging import logging

logger = logging.getLogger(__name__)


def chat(messages: list[ChatMessage],
         model: str = DEFAULT_CHAT_MODEL,
         system_message: str = None,
         temperature: float = 0.0,
         tools: dict[str, dict] = ENABLED_TOOLS,
         tool_choice: Literal['auto', 'none'] = 'auto') -> object:
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
        chat_frame=json.dumps([f.dict() for f in chat_frame[:-1]]).encode('utf-8'),
        content=new_chat_message.content.encode('utf-8'),
        role=new_chat_message.role,
    )
    with SessionLocal() as session:
        session.add(message_received)
        session.commit()
        session.refresh(message_received)

    # Do completion request
    response = CHAT_COMPLETION_FUNCTIONS[model](chat_frame,
                                                system_message=system_message,
                                                temperature=temperature,
                                                tools=tools,
                                                tool_choice=tool_choice)
    """
    $ curl -s localhost:8001/chat \
        -H 'content-type: application/json' \
        -X POST -d '{"messages": [{"role": "user", "content": "Hello"}]}'
    {
      "id": "chatcmpl-9XLdhn7SsScrsPtEa1SNg37BxcH4M",
      "choices": [
        {
          "finish_reason": "stop",
          "index": 0,
          "logprobs": null,
          "message": {
            "content": "Hello! How can I assist you today?",
            "role": "assistant",
            "function_call": null,
            "tool_calls": null
          }
        }
      ],
      "created": 1717735033,
      "model": "gpt-4-0613",
      "object": "chat.completion",
      "system_fingerprint": null,
      "usage": {
        "completion_tokens": 9,
        "prompt_tokens": 8,
        "total_tokens": 17
      }
    }
    """
    new_response = response.choices[0].message
    logger.debug("response: %s", new_response.content)

    # If a tool should be used, call it.
    if new_response.tool_calls:
        tool_args = json.loads(new_response.tool_calls[0].function.arguments)
        new_response.content = tool_wrapper(new_response.tool_calls[0].function.name, tool_args)

    ##
    # TODO: post-response >>HERE<<

    # Update message history
    message_replied = MessageReplied(
        content=None if new_response.tool_calls else new_response.content.encode('utf-8'),
        message_received_id=message_received.id,
        role=new_response.role,
        tool_func_name=None if not new_response.tool_calls else new_response.tool_calls[0].function.name
    )
    with SessionLocal() as session:
        session.add(message_replied)
        session.commit()
        session.refresh(message_replied)
    return {
        "message_received_id": message_received.id,
        "message_replied_id": message_replied.id,
        "response": response,
    }
