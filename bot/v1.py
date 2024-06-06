from openai import OpenAI
from pydantic import BaseModel
from typing import List
import faiss
import tiktoken

from bot.db_chat_history import SessionLocal as ChatHistorySessionLocal
from bot.vector_store import OpenAITextEmbedding3SmallDim1536


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]


class OpenAIChatBot:
    def __init__(self, model="gpt-4o"):
        self.default_vector_store = OpenAITextEmbedding3SmallDim1536()  # TODO: Dynamically explore topical vector DBs?
        self.enc = tiktoken.encoding_for_model(model)
        self.model = model

    def chat(self, messages: list[ChatMessage]) -> object:
        total_tokens = 0
        new_messages = []
        for chat_message in reversed(messages):  # In reverse because message list is ordered from oldest to newest.
            message = chat_message.model_dump()
            message_tokens = len(self.enc.encode(message['content']))
            # If adding `message_tokens` pushes us over the limit, stop appending
            if message_tokens + total_tokens > self.enc.max_token_value:
                break
            total_tokens += message_tokens
            new_messages.append(message)
        response = OpenAI().chat.completions.create(
            model=self.model,
            messages=reversed(new_messages)  # Undo reverse
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

        #self.default_vector_store.add(message, response.choices.pop().message.content)
        #self.chat_history.append((sender, message, response.choices.pop().message.content))
        #return response.choices.pop().message.content
        return response
