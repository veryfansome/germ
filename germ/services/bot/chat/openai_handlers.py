from abc import ABC, abstractmethod
from datetime import datetime
from openai.types.chat.chat_completion import ChatCompletion
from typing import Optional
import asyncio
import json
import logging

from germ.api.models import ChatRequest, ChatResponse
from germ.services.bot.chat import async_openai_client, openai_beta
from germ.services.bot.websocket import WebSocketReceiveEventHandler, WebSocketSender
from germ.observability.annotations import measure_exec_seconds
from germ.settings.openai_settings import (CHAT_MODEL, MINI_MODEL, REASONING_MODEL, ROUTING_MODEL,
                                           ENABLED_CHAT_MODELS, ENABLED_IMAGE_MODELS, HTTPX_TIMEOUT)

logger = logging.getLogger(__name__)


##
# User facing routing and routable chat handlers


class RoutableChatEventHandler(WebSocketReceiveEventHandler, ABC):

    @abstractmethod
    def get_function_name(self):
        pass

    @abstractmethod
    def get_function_settings(self):
        pass


class ChatModelEventHandler(RoutableChatEventHandler):
    def __init__(self, model: str = CHAT_MODEL):
        self.function_name = f"use_{model}"
        self.function_settings = {
            "type": "function",
            "function": {
                "name": self.function_name,
                "description": (f"Use {model} to generate a better zero-shot response "
                                f"or if the user specifically asks for {model}."),
                "parameters": {},
            },
        }
        self.model = model

    async def do_chat_completion(self, chat_request: ChatRequest) -> ChatCompletion:
        return await async_openai_client.chat.completions.create(
            messages=[message.model_dump() for message in chat_request.messages] + [
                {"role": "system",
                 "content": ("Answer in valid Markdown format only. "
                             "Don't use code blocks unnecessarily but always use them when dealing with code.")}
            ],
            model=self.model,
            n=1, timeout=HTTPX_TIMEOUT)

    def get_function_name(self):
        return self.function_name

    def get_function_settings(self):
        return self.function_settings

    @measure_exec_seconds(use_logging=True, use_prometheus=True)
    async def on_receive(self, user_id: int, conversation_id: int, dt_created: datetime, text_sig: str,
                         chat_request: ChatRequest, ws_sender: WebSocketSender):
        completion = await self.do_chat_completion(chat_request)
        await ws_sender.send_reply(
            dt_created,
            ChatResponse(complete=True, content=completion.choices[0].message.content, model=completion.model)
        )


class ImageModelEventHandler(RoutableChatEventHandler):
    def __init__(self, model: str, httpx_timeout: float = 90):
        self.function_name = f"use_{model}"
        self.function_settings = {
            "type": "function",
            "function": {
                "name": self.function_name,
                "description": " ".join((
                    f"Use {model} to generate an image.",
                )),
                "parameters": {},
            },
        }
        self.httpx_timeout = httpx_timeout
        self.model = model

    async def generate_markdown_image(self, chat_request: ChatRequest):
        image_model_inputs = await generate_image_model_inputs(chat_request)
        try:
            submission = await async_openai_client.images.generate(
                prompt=image_model_inputs['prompt'], model=self.model, n=1,
                size=image_model_inputs['size'], style=image_model_inputs['style'],
                timeout=HTTPX_TIMEOUT)
            return f"[![{image_model_inputs['prompt']}]({submission.data[0].url})]({submission.data[0].url})"
        except Exception as e:
            logger.error(e)
            return f"[![Failed to generate image](/static/assets/oops_image.jpg)](/static/assets/oops_image.jpg)"

    def get_function_name(self):
        return self.function_name

    def get_function_settings(self):
        return self.function_settings

    @measure_exec_seconds(use_logging=True, use_prometheus=True)
    async def on_receive(self, user_id: int, conversation_id: int, dt_created: datetime, text_sig: str,
                         chat_request: ChatRequest, ws_sender: WebSocketSender):
        markdown_image = await self.generate_markdown_image(chat_request)
        _ = asyncio.create_task(
            ws_sender.send_reply(
                dt_created,
                ChatResponse(complete=True, content=markdown_image, model=self.model)
            )
        )


class ReasoningChatModelEventHandler(RoutableChatEventHandler):
    def __init__(self, assistant_helper: openai_beta.AssistantHelper = None,
                 model: str = REASONING_MODEL, httpx_timeout: float = 180):
        self.assistant_helper = assistant_helper
        self.function_name = f"use_{model}"
        self.function_settings = {
            "type": "function",
            "function": {
                "name": self.function_name,
                "description": (f"Use {model} to generate a better zero-shot response with reasoning "
                                f"or if the user specifically asks for {model}."),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reasoning_effort": {
                            "type": "string",
                            "description": "Guidance on how many reasoning tokens to generate before responding.",
                            "enum": ["low", "medium", "high"],
                        },
                    },
                    "required": ["reasoning_effort"]
                },
            },
        }
        self.httpx_timeout = httpx_timeout
        self.model = model

    async def do_chat_completion(self, chat_request: ChatRequest) -> ChatCompletion:
        return await async_openai_client.chat.completions.create(
            messages=[message.model_dump() for message in chat_request.messages] + [
                {"role": "system",
                 "content": ("Answer in valid Markdown format only. "
                             "Don't use code blocks unnecessarily but always use them when dealing with code.")}
            ],
            model=self.model,
            n=1, timeout=self.httpx_timeout,
            **chat_request.parameters,
        )

    def get_function_name(self):
        return self.function_name

    def get_function_settings(self):
        return self.function_settings

    @measure_exec_seconds(use_logging=True, use_prometheus=True)
    async def on_receive(self, user_id: int, conversation_id: int, dt_created: datetime, text_sig: str,
                         chat_request: ChatRequest, ws_sender: WebSocketSender):
        completion = await self.do_chat_completion(chat_request)
        await ws_sender.send_reply(
            dt_created,
            ChatResponse(
                complete=True,
                content=completion.choices[0].message.content,
                model=f"{completion.model}[{'|'.join([f'{k}:{v}' for k, v in chat_request.parameters.items()])}]"
            )
        )


class ChatRoutingEventHandler(ChatModelEventHandler):
    """
    This class is a user facing router. It returns simple completions and delegates to other handlers
    if the situation calls for other tools.
    """

    def __init__(self, model: str = ROUTING_MODEL, assistant_helper: openai_beta.AssistantHelper = None):
        super().__init__(model)
        self.assistant_helper = assistant_helper
        self.model: str = model
        self.tools: dict[str, RoutableChatEventHandler] = {}

        tools: dict[str, RoutableChatEventHandler] = {}
        for m in ENABLED_CHAT_MODELS:
            if m.startswith("gpt-"):
                tools[m] = ChatModelEventHandler(model=m)
            elif m.startswith("o1") or m.startswith("o3"):
                tools[m] = ReasoningChatModelEventHandler(assistant_helper=assistant_helper, model=m)
        for m in ENABLED_IMAGE_MODELS:
            tools[m] = ImageModelEventHandler(m)
        for tool in tools.values():
            self.tools[tool.get_function_name()] = tool
            logger.info(f"Added {tool.get_function_name()} => {tool}")

    @measure_exec_seconds(use_logging=True, use_prometheus=True)
    async def on_receive(self, user_id: int, conversation_id: int, dt_created: datetime, text_sig: str,
                         chat_request: ChatRequest, ws_sender: WebSocketSender):
        if chat_request.uploaded_filenames:
            logger.info(f"Uploaded file: {chat_request.uploaded_filenames}")
            await self.assistant_helper.handle_in_a_thread(
                conversation_id, dt_created, chat_request, ws_sender
            )
        else:
            completion = await self.do_chat_completion(chat_request)
            error = False
            if completion is None:
                await ws_sender.send_message(
                    ChatResponse(complete=True, content="Sorry, I'm unable to access my language model.", error=True)
                )
            elif completion.choices[0].message.content is None:
                if completion.choices[0].message.tool_calls is not None:
                    tool_response_tasks = []
                    for tool_call in completion.choices[0].message.tool_calls:
                        chat_request.parameters = json.loads(tool_call.function.arguments)
                        tool_response_tasks.append(
                            asyncio.create_task(
                                self.tools[tool_call.function.name].on_receive(
                                    user_id, conversation_id, dt_created, text_sig, chat_request, ws_sender
                                )
                            )
                        )
                    # Let user know you're delegating
                    tool_names = [f"`{t.function.name}`" for t in completion.choices[0].message.tool_calls]
                    await ws_sender.send_message(
                        ChatResponse(complete=False, content=f"One moment, using tools: {''.join(tool_names)}.")
                    )
                    # TODO: Tools calls may not mean responded going forward
                    await asyncio.gather(*tool_response_tasks)
                    return
                else:
                    logger.error("Completion content and tool_calls are both missing.", completion)
                    completion.choices[0].message.content = "Something went wrong. I don't have a response."
                    completion.model = "none"
                    error = True
            # Return completed response
            await ws_sender.send_reply(
                dt_created,
                ChatResponse(complete=True, content=completion.choices[0].message.content,
                             error=error, model=completion.model)
            )

    async def do_chat_completion(self, chat_request: ChatRequest) -> Optional[ChatCompletion]:
        tools = [t.get_function_settings() for t in self.tools.values()]
        try:
            return await async_openai_client.chat.completions.create(
                messages=[message.model_dump() for message in chat_request.messages] + [
                    {"role": "system",
                     "content": "Reply if trivial or use a single more appropriate tool."},
                    {"role": "system",
                     "content": ("Answer in valid Markdown format only. "
                                 "Don't use code blocks unnecessarily but always use them when dealing with code.")},
                    {"role": "system",
                     "content": "Always use a tool when dealing with programming or highly technical topics."},
                ],
                model=self.model,
                n=1, tools=tools,
                timeout=HTTPX_TIMEOUT)
        except Exception as e:
            logger.error(e)
        return None


##
# Functions


@measure_exec_seconds(use_logging=True, use_prometheus=True)
async def generate_image_model_inputs(chat_request: ChatRequest):
    completion = await async_openai_client.chat.completions.create(
        messages=([{
            "role": "system",
            "content": "The user wants an image."
        }] + [message.model_dump() for message in chat_request.messages]),

        model=MINI_MODEL, n=1,
        tool_choice={
            "type": "function",
            "function": {"name": "generate_image"}
        },
        tools=[{
            "type": "function",
            "function": {
                "name": "generate_image",
                "description": "Return a generated image, given a text prompt, image size, and style.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "The text prompt for the image model."
                        },
                        "size": {
                            "type": "string",
                            "enum": ["1024x1024", "1024x1792", "1792x1024"],
                        },
                        "style": {
                            "type": "string",
                            "enum": ["natural", "vivid"],
                        },
                    },
                    "required": ["prompt", "size", "style"],
                    "additionalProperties": False,
                },
            }
        }],
        timeout=HTTPX_TIMEOUT)
    return json.loads(completion.choices[0].message.tool_calls[0].function.arguments)
