from abc import ABC, abstractmethod
from openai import OpenAI
from openai.types.chat.chat_completion import ChatCompletion
from sqlalchemy import func as sql_func
from starlette.concurrency import run_in_threadpool
from typing import Optional

from typing import Iterable, Awaitable
import asyncio
import json
import logging

from api.models import ChatRequest, ChatResponse
from bot.websocket import SessionMonitor, WebSocketEventHandler, WebSocketSender
from db.models import (ChatSession, ChatSessionChatUserLink, ChatSessionChatUserProfileLink,
                       ChatUser, ChatUserProfile, SessionLocal)
from observability.annotations import measure_exec_seconds
from settings.openai_settings import (DEFAULT_CHAT_MODEL, DEFAULT_MINI_MODEL, DEFAULT_ROUTING_MODEL,
                                      ENABLED_CHAT_MODELS, ENABLED_IMAGE_MODELS, HTTPX_TIMEOUT)

logger = logging.getLogger(__name__)


class BackgroundChatEventHandler(WebSocketEventHandler, ABC):

    @classmethod
    async def run_in_background(cls, target_tasks: Iterable[Awaitable]):
        for target_task in target_tasks:
            task = asyncio.create_task(target_task)


class RoutableChatEventHandler(WebSocketEventHandler, ABC):

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
                    f"or if {model} would return a better result than one generated by {DEFAULT_ROUTING_MODEL}.",
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


def messages_to_transcript(chat_request: ChatRequest):
    """
    GPT-4o recommended the following format because it captures directionality and provides clear boundaries to
    messages, which is helpful when trying to get embeddings that capture the context of the conversation.

    [USER] Hi! [ASSISTANT]
    [ASSISTANT] Hey! How can I help you? [USER]
    [USER] Tell me a joke [ASSISTANT]

    :param chat_request:
    :return: transcript as a string
    """
    transcript_lines = []
    for message in chat_request.messages:
        end_tag = "user" if message.role == "assistant" else "assistant"
        transcript_lines.append(f"[{message.role.upper()}] {message.content} [{end_tag.upper()}]")
    return "\n".join(transcript_lines)


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


class SessionMonitoringHandler(SessionMonitor):
    def __init__(self):
        pass

    @measure_exec_seconds(use_logging=True, use_prometheus=True)
    async def on_tick(self, chat_session_id: int, ws_sender: WebSocketSender):
        pass


class UserIdentifyingHandler(WebSocketEventHandler):
    def __init__(self, user_profile, model: str = DEFAULT_MINI_MODEL):
        self.model = model
        self.user_profile = user_profile

    @measure_exec_seconds(use_logging=True, use_prometheus=True)
    async def on_receive(self, chat_session_id: int, chat_request_received_id: id,
                         chat_request: ChatRequest, ws_sender: WebSocketSender):
        session_users = await run_in_threadpool(self.get_session_users, chat_session_id)
        # TODO: Update chat_message_idea table
        if session_users:
            pass  # TODO: after long delays, check if user is the same
        else:
            top_users = await run_in_threadpool(self.get_top_users)
            if top_users:
                pass  # TODO: guess if top users
            #elif "user_first_name" not in self.user_profile['profile'] and "user_last_name" not in self.user_profile['profile']:
            #    pass  # TODO: ask for identification
            #else:
            #    pass  # TODO: does partial name match any known users? Create new user unless matched.

    @classmethod
    def get_top_users(cls, limit: int = 2):
        top_users = []
        with SessionLocal() as session:
            chat_users_records = (
                session.query(ChatUser)
                .join(ChatSessionChatUserLink)
                .join(ChatSession)
                .group_by(ChatUser.chat_user_id)
                .order_by(sql_func.count(ChatSession.chat_session_id).desc())
                .limit(limit)
                .all()
            )
            for idx, user_record in enumerate(chat_users_records):
                top_users[idx] = {
                    "user_id": user_record.chat_user_id,
                    "user_first_name": user_record.chat_user_first_name,
                    "user_middle_name_or_initials": user_record.chat_user_middle_name_or_initials,
                    "user_last_name": user_record.chat_user_last_name,
                }
            return top_users

    @classmethod
    def get_session_users(cls, chat_session_id: int):
        session_users = []
        with SessionLocal() as session:
            chat_session = session.query(ChatSession).filter_by(chat_session_id=chat_session_id).first()
            if not chat_session:
                logger.error(f"chat_session_id {chat_session_id} not found")
                return session_users
            for idx, user_record in enumerate(chat_session.chat_users):
                session_users[idx] = {
                    "user_id": user_record.chat_user_id,
                    "user_first_name": user_record.chat_user_first_name,
                    "user_middle_name_or_initials": user_record.chat_user_middle_name_or_initials,
                    "user_last_name": user_record.chat_user_last_name,
                }
            return session_users


class UserProfilingHandler(BackgroundChatEventHandler):
    """
    This class is an off-stage router. It analyzes conversations to construct shifting profiles of users, which
    are persisted and linked with sessions. As more complete profiles of users are constructed, additional handlers
    can be used to if the situation calls for it.
    """

    tool_func_name = "update_user_profile"
    tool = {
        "type": "function",
        "function": {
            "name": tool_func_name,
            "description": " ".join((
                "Update the user's profile with data extracted from the conversation.",
            )),
            "parameters": {
                "type": "object",
                "properties": {
                    "conversation_topics": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "User topics from the conversation.",
                    },
                    "user_facts": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Facts about the user from the conversation.",
                    },
                    "user_intention": {
                        "type": "string",
                        "description": "User intention of the conversation.",
                    },
                    "user_opinions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "User opinions from the conversation.",
                    },
                    "user_attitude_toward_assistant": {
                        "type": "number",
                        "description": " ".join((
                            "On a scale of 0(very bad) to 5(very good),",
                            "how the user feels about the assistant.",
                        )),
                    },
                    "user_attitude_toward_self": {
                        "type": "number",
                        "description": " ".join((
                            "On a scale of 0(very bad) to 5(very good),",
                            "how the user feels about themself.",
                        )),
                    },
                    "user_tone": {
                        "type": "string",
                        "description": "User tone from the conversation.",
                    },
                    "user_honorific": {
                        "type": "string",
                        "description": "User's title or honorific.",
                    },
                    "user_first_name": {
                        "type": "string",
                        "description": "User's first name.",
                    },
                    "user_middle_name_or_initials": {
                        "type": "string",
                        "description": "User's middle name or initials.",
                    },
                    "user_last_name": {
                        "type": "string",
                        "description": "User's last name.",
                    },
                },
                "required": ["user_intention", "user_tone"],
                "additionalProperties": False,
            },
        },
    }

    def __init__(self, model: str = DEFAULT_CHAT_MODEL):
        self.model = model

    @measure_exec_seconds(use_logging=True, use_prometheus=True)
    async def on_receive(self, chat_session_id: int, chat_request_received_id: id,
                         chat_request: ChatRequest, ws_sender: WebSocketSender):
        task = asyncio.create_task(
            run_in_threadpool(self.process_chat_request,
                              chat_session_id, chat_request_received_id, chat_request, ws_sender))

    def process_chat_request(self, chat_session_id: int, chat_request_received_id: id,
                             chat_request: ChatRequest, ws_sender: WebSocketSender):
        with OpenAI() as client:
            completion = client.chat.completions.create(
                messages=[message.model_dump() for message in chat_request.messages if message.role != "system"],
                model=self.model,
                n=1, temperature=0,
                tool_choice={
                    "type": "function",
                    "function": {"name": UserProfilingHandler.tool_func_name}
                },
                tools=[UserProfilingHandler.tool],
                timeout=HTTPX_TIMEOUT)
            user_profile = {
                "profile": completion.choices[0].message.tool_calls[0].function.arguments,
                "system_messages": [message.model_dump() for message in chat_request.messages if message.role == "system"]
            }
            logger.info(user_profile)
            with SessionLocal() as session:
                user_profile_record = ChatUserProfile(chat_user_profile=user_profile)
                session.add(user_profile_record)
                session.commit()
                user_profile_to_session_link = ChatSessionChatUserProfileLink(
                    chat_session_id=chat_session_id, chat_user_profile_id=user_profile_record.chat_user_profile_id)
                session.add(user_profile_to_session_link)
                session.commit()
            asyncio.run(self.run_in_background([
                UserIdentifyingHandler(user_profile).on_receive(
                    chat_session_id, chat_request_received_id, chat_request, ws_sender),
            ]))
