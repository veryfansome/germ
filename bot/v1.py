from openai import OpenAI
from pydantic import BaseModel
import json
import tiktoken

from bot.db_chat_history import MessageReceived, MessageSent, SessionLocal as ChatHistorySessionLocal
from bot.logging_config import logging
from bot.vector_store import OpenAITextEmbedding3SmallDim1536

logger = logging.getLogger(__name__)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]


class OpenAIChatBot:
    def __init__(self, default_model="gpt-4o"):
        self.default_model = default_model
        self.default_vector_store = OpenAITextEmbedding3SmallDim1536()  # TODO: Dynamically explore topical vector DBs?
        self.enabled_models = [self.default_model, "dall-e-3"]
        self.enabled_vector_stores = [self.default_vector_store]

    def chat(self, messages: list[ChatMessage]) -> object:
        new_chat_message: ChatMessage = messages[-1]
        # Insert received message
        logger.info("received: %s", new_chat_message.content)

        # Trim message list to avoid hitting selected model's token limit.
        enc = tiktoken.encoding_for_model(self.default_model)
        total_tokens = 0
        reversed_chat_frame = []
        for chat_message in reversed(messages):  # In reverse because message list is ordered from oldest to newest.
            message_dict = chat_message.model_dump()
            message_tokens = len(enc.encode(message_dict['content']))
            # If adding `message_tokens` pushes us over the limit, stop appending
            if message_tokens + total_tokens > enc.max_token_value:
                break
            total_tokens += message_tokens
            reversed_chat_frame.append(message_dict)
        chat_frame = tuple(reversed(reversed_chat_frame))  # Undo previously reversed order

        # Update message history
        message_received_session = ChatHistorySessionLocal()
        message_received = MessageReceived(
            chat_frame=json.dumps(chat_frame[:-1]).encode('utf-8'),
            content=new_chat_message.content.encode('utf-8'),
            role=new_chat_message.role,
        )
        message_received_session.add(message_received)
        message_received_session.commit()

        # Do completion request
        response = OpenAI().chat.completions.create(
            messages=chat_frame,
            model=self.default_model,
        )
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

        # self.default_vector_store.add(message, response.choices.pop().message.content)
        # self.chat_history.append((sender, message, response.choices.pop().message.content))

        # Update message history
        message_received_session.refresh(message_received)
        new_response = response.choices[0].message
        logger.info("response: %s", new_response.content)
        response_session = ChatHistorySessionLocal()
        response_session.add(
            MessageSent(
                content=new_response.content.encode('utf-8'),
                message_received_id=message_received.id,
            )
        )
        response_session.commit()
        return response
