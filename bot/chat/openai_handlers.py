from abc import ABC, abstractmethod
from openai import OpenAI
from openai.types.chat.chat_completion import ChatCompletion
from starlette.concurrency import run_in_threadpool
from typing import Callable, Optional
import asyncio
import json
import logging
import mistune

from api.models import ChatRequest, ChatResponse
from bot.graph.idea import idea_graph
from bot.lang.markdown_extractor import MarkdownExtractor
from bot.websocket import WebSocketReceiveEventHandler, WebSocketSendEventHandler, WebSocketSender
from observability.annotations import measure_exec_seconds
from settings.openai_settings import (DEFAULT_CHAT_MODEL, DEFAULT_MINI_MODEL, DEFAULT_ROUTING_MODEL,
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
    def __init__(self, model: str = DEFAULT_CHAT_MODEL):
        self.function_name = f"use_{model}"
        self.function_settings = {
            "type": "function",
            "function": {
                "name": self.function_name,
                "description": " ".join((
                    f"Use {model} to generate a reply if the user asks for {model}",
                    f"or if conversation is not banter.",
                )),
                "parameters": {},
            },
        }
        self.model = model

    def do_chat_completion(self, chat_request: ChatRequest) -> ChatCompletion:
        with OpenAI() as client:
            return client.chat.completions.create(
                messages=[message.model_dump() for message in chat_request.messages],
                model=self.model,
                n=1, temperature=chat_request.temperature,
                timeout=HTTPX_TIMEOUT)

    def get_function_name(self):
        return self.function_name

    def get_function_settings(self):
        return self.function_settings

    @measure_exec_seconds(use_logging=True, use_prometheus=True)
    async def on_receive(self,
                         chat_session_id: int, chat_request_received_id: int,
                         chat_request: ChatRequest, ws_sender: WebSocketSender):
        completion = await run_in_threadpool(self.do_chat_completion, chat_request)
        task = asyncio.create_task(ws_sender.return_chat_response(
            chat_request_received_id, ChatResponse(
                content=completion.choices[0].message.content,
                model=completion.model)))


class ImageModelEventHandler(RoutableChatEventHandler):
    def __init__(self, model: str):
        self.function_name = f"use_{model}"
        self.function_settings = {
            "type": "function",
            "function": {
                "name": self.function_name,
                "description": " ".join((
                    f"Use {model} to generate an image if {model} is likely to achieve good results.",
                )),
                "parameters": {},
            },
        }
        self.model = model

    def generate_markdown_image(self, chat_request: ChatRequest):
        image_model_inputs = generate_image_model_inputs(chat_request)
        try:
            with OpenAI() as client:
                submission = client.images.generate(
                    prompt=image_model_inputs['prompt'], model=self.model, n=1,
                    size=image_model_inputs['size'], style=image_model_inputs['style'],
                    timeout=HTTPX_TIMEOUT)
                return f"[![{image_model_inputs['prompt']}]({submission.data[0].url})]({submission.data[0].url})"
        except Exception as e:
            logger.error(e)
            return f"[![Failed to generate image](/static/oops_image.jpg)](/static/oops_image.jpg)"

    def get_function_name(self):
        return self.function_name

    def get_function_settings(self):
        return self.function_settings

    @measure_exec_seconds(use_logging=True, use_prometheus=True)
    async def on_receive(self,
                         chat_session_id: int, chat_request_received_id: int,
                         chat_request: ChatRequest, ws_sender: WebSocketSender):
        markdown_image = await run_in_threadpool(self.generate_markdown_image, chat_request)
        task = asyncio.create_task(
            ws_sender.return_chat_response(
                chat_request_received_id,
                ChatResponse(content=markdown_image, model=self.model)))


USER_FACING_CHAT_HANDLERS: dict[str, RoutableChatEventHandler] = {}
for m in ENABLED_CHAT_MODELS:
    USER_FACING_CHAT_HANDLERS[m] = ChatModelEventHandler(m)
for m in ENABLED_IMAGE_MODELS:
    USER_FACING_CHAT_HANDLERS[m] = ImageModelEventHandler(m)


class ChatRoutingEventHandler(ChatModelEventHandler):
    """
    This class is a user facing router. It returns simple completions and delegates to other handlers
    if the situation calls for other tools.
    """

    def __init__(self, model: str = DEFAULT_ROUTING_MODEL):
        super().__init__(model)
        self.model: str = model
        self.tools: dict[str, RoutableChatEventHandler] = {}
        for chat_handler in USER_FACING_CHAT_HANDLERS.values():
            self.add_tool(chat_handler.get_function_name(), chat_handler)
            logger.info(f"added {chat_handler.get_function_name()} => {chat_handler}")

    def add_tool(self, tool_name: str, chat_handler: RoutableChatEventHandler):
        self.tools[tool_name] = chat_handler

    @measure_exec_seconds(use_logging=True, use_prometheus=True)
    async def on_receive(self,
                         chat_session_id: int, chat_request_received_id: int,
                         chat_request: ChatRequest, ws_sender: WebSocketSender):
        completion = await run_in_threadpool(self.do_chat_completion, chat_request)
        if completion is None:
            task = asyncio.create_task(ws_sender.return_chat_response(
                chat_request_received_id,
                ChatResponse(content="Sorry, I'm unable to access my language model.")))
            return
        elif completion.choices[0].message.content is None:
            if completion.choices[0].message.tool_calls is not None:
                for tool_call in completion.choices[0].message.tool_calls:
                    task = asyncio.create_task(
                        self.tools[tool_call.function.name].on_receive(
                            chat_session_id, chat_request_received_id, chat_request, ws_sender
                        )
                    )
                return
            else:
                logger.error("completion content and tool_calls are both missing", completion)
                completion.choices[0].message.content = "Strange... I don't have a response"
        task = asyncio.create_task(ws_sender.return_chat_response(
            chat_request_received_id,
            ChatResponse(
                content=completion.choices[0].message.content,
                model=completion.model)))

    def do_chat_completion(self, chat_request: ChatRequest) -> Optional[ChatCompletion]:
        tools = [t.get_function_settings() for t in self.tools.values()]
        try:
            with OpenAI() as client:
                return client.chat.completions.create(
                    messages=[message.model_dump() for message in chat_request.messages],
                    model=self.model,
                    n=1, temperature=chat_request.temperature,
                    tools=tools,
                    timeout=HTTPX_TIMEOUT)
        except Exception as e:
            logger.error(e)
        return None


##
# Internal chat handlers


class ResponseGraphingHandler(WebSocketSendEventHandler):
    def __init__(self):
        self.node_types = {
            "code_block": "CodeBlock",
            "header": "Header",
            "paragraph": "Paragraph",
        }

    async def on_send(self,
                      chat_response_sent_id: int,
                      chat_response: ChatResponse,
                      chat_session_id: int,
                      chat_request_received_id: int = None):
        elements = await run_in_threadpool(convert_chat_response_to_markdown_elements, chat_response)
        """
        ('header', 1, 'Heading 1')
        ('paragraph', 'This is a paragraph with some text.')
        ('header', 2, 'Heading 2')
        ('list', 'unordered', '- Bullet point 1\n- Bullet point 2\n')
        ('list_item', 'Bullet point 1')
        ('list_item', 'Bullet point 2')
        ('list', 'ordered', '1. Ordered item 1\n2. Ordered item 2\n')
        ('list_item', 'Ordered item 1')
        ('list_item', 'Ordered item 2')
        ('code_block', None, 'def hello_world():\n    print("Hello, world!")\n')
        """
        last_element_type = None
        last_element_attrs = None
        link_attrs = {
            "chat_request_received_id": chat_request_received_id,
            "chat_response_sent_id": chat_response_sent_id,
            "chat_session_id": chat_session_id,
        }
        for element in elements:
            logger.debug(f"markdown element: {element}")
            if element[0] == "paragraph":
                _, paragraph_id, _ = await idea_graph.add_paragraph(element[1])
                this_element_attrs = {"paragraph_id": paragraph_id}
            elif element[0] == "code_block":
                _, code_block_id, _ = await idea_graph.add_code_block(element[2], language=element[1])
                this_element_attrs = {"code_block_id": code_block_id}
            elif element[0] == "header":
                await idea_graph.add_header(element[1])
                this_element_attrs = {"text": element[1]}
            #elif element[0] == "list":
            #    pass
            #elif element[0] == "list_item":
            #    pass
            else:
                logger.info(f"unsupported element type: {element[0]}")
                continue

            _ = asyncio.create_task(idea_graph.link_page_element_to_chat_response(
                self.node_types[element[0]], this_element_attrs, chat_response_sent_id))

            if last_element_type is not None and last_element_attrs is not None:
                _ = asyncio.create_task(idea_graph.link_successive_page_elements(
                    last_element_type, last_element_attrs,
                    self.node_types[element[0]], this_element_attrs,
                    link_attrs))

            last_element_attrs = this_element_attrs
            last_element_type = self.node_types[element[0]]


class UserProfileConsumer(ABC):
    @abstractmethod
    async def on_new_user_profile(self, user_profile, chat_session_id: int, chat_request_received_id: id,
                                  chat_request: ChatRequest, ws_sender: WebSocketSender):
        pass


class UserProfilingHandler(WebSocketReceiveEventHandler):
    def __init__(self, tool_properties_spec,
                 consumers: list[UserProfileConsumer] = None,
                 model: str = DEFAULT_CHAT_MODEL,
                 post_func: Callable[[str], str] = None,
                 tool_func_description: str = "",
                 tool_func_name: str = "",
                 tool_parameter_description: str = ""):
        self.consumers: list[UserProfileConsumer] = consumers if consumers else []
        self.model = model
        self.post_func = post_func
        self.tool_properties_spec = tool_properties_spec
        self.tool_func_name = tool_func_name if tool_func_name else "store_user_profile"
        self.tool_func_description = (tool_func_description
                                      if tool_func_description
                                      else "Store user profile generated from an analysis of the conversation.")
        self.tool_parameter_description = (tool_parameter_description
                                           if tool_parameter_description
                                           else "Analysis of the user and the conversation to store.")
        self.tool = {
            "type": "function",
            "function": {
                "name": self.tool_func_name,
                "description": self.tool_func_description,
                "parameters": {
                    "type": "object",
                    "description": self.tool_parameter_description,
                    "properties": tool_properties_spec,
                    "required": list(tool_properties_spec.keys()),
                    "additionalProperties": False,
                },
            },
        }

    def add_consumer(self, consumer: UserProfileConsumer):
        self.consumers.append(consumer)

    @measure_exec_seconds(use_logging=True, use_prometheus=True)
    async def on_receive(self, chat_session_id: int, chat_request_received_id: id,
                         chat_request: ChatRequest, ws_sender: WebSocketSender):
        user_profile = await run_in_threadpool(self.process_chat_request, chat_session_id, chat_request)
        processed_profile = {}
        for profile_name, profile_text in user_profile["profile"].items():
            profile_text = self.post_func(profile_text)
            processed_profile[profile_name] = profile_text
            _, sentence_id, _ = await idea_graph.add_sentence(profile_text)
            _ = asyncio.create_task(
                idea_graph.link_reactive_sentence_to_chat_request(chat_request_received_id, sentence_id))
            # Get how many nodes are linked to session
            # If first node, create starts link
        user_profile["profile"] = processed_profile
        for consumer in self.consumers:
            task = asyncio.create_task(
                consumer.on_new_user_profile(
                    user_profile, chat_session_id, chat_request_received_id, chat_request, ws_sender))

    def process_chat_request(self, chat_session_id: int, chat_request: ChatRequest):
        with OpenAI() as client:
            completion = client.chat.completions.create(
                messages=[message.model_dump() for message in chat_request.messages if message.role != "system"],
                model=self.model,
                n=1, temperature=0,
                tool_choice={
                    "type": "function",
                    "function": {"name": self.tool_func_name}
                },
                tools=[self.tool],
                timeout=HTTPX_TIMEOUT)
            profile_parameters = json.loads(completion.choices[0].message.tool_calls[0].function.arguments)
            user_profile = {
                "chat_session_id": chat_session_id,
                "system_messages": [
                    message.model_dump() for message in chat_request.messages if message.role == "system"],
                "messages": chat_request.messages,
                "profile": profile_parameters,
            }
            return user_profile


##
# Functions


def convert_chat_response_to_markdown_elements(chat_response: ChatResponse):
    extractor = MarkdownExtractor()
    mistune.create_markdown(renderer=extractor)(chat_response.content)
    return extractor.elements


@measure_exec_seconds(use_logging=True, use_prometheus=True)
def generate_image_model_inputs(chat_request: ChatRequest):
    with OpenAI() as client:
        completion = client.chat.completions.create(
            messages=([{
                "role": "system",
                "content": "The user wants an image."
            }] + [message.model_dump() for message in chat_request.messages]),

            model=DEFAULT_MINI_MODEL, n=1, temperature=chat_request.temperature,
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
