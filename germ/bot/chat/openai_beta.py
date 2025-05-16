from datetime import datetime
from openai.lib.streaming import AsyncAssistantEventHandler
from pathlib import Path
from traceback import format_exc
from typing_extensions import override
import asyncio
import logging.config

from germ.api.models import ChatRequest, ChatResponse
from germ.bot.chat import async_openai_client
from germ.bot.websocket import WebSocketSender
from germ.observability.annotations import measure_exec_seconds
from germ.settings import openai_settings

logger = logging.getLogger(__name__)


class AssistantHelper:
    def __init__(self):
        self.assistants = {}
        self.stored_files = {}
        self.threads = {}

    async def create_or_update_assistant(self, name: str, **kwargs):
        if name not in self.assistants:
            asst = await async_openai_client.beta.assistants.create(
                name=name,
                **kwargs,
            )
            self.assistants[asst.name] = asst
        else:
            asst = await async_openai_client.beta.assistants.update(
                self.assistants[name].id,
                **kwargs,
            )
            self.assistants[asst.name] = asst
        logger.info(f"initialized {asst}")

    @measure_exec_seconds(use_logging=True, use_prometheus=True)
    async def handle_in_a_thread(self, conversation_id: int, dt_created: datetime,
                                 chat_request: ChatRequest, ws_sender: WebSocketSender):
        if chat_request.uploaded_filenames:
            for filename in chat_request.uploaded_filenames:
                if filename not in self.stored_files:
                    new_stored_file = await async_openai_client.files.create(
                        file=Path(f"/tmp/{filename}"),
                        purpose='assistants',
                        timeout=openai_settings.HTTPX_TIMEOUT,
                    )
                    self.stored_files[filename] = new_stored_file
                    logger.info(f"uploaded: {new_stored_file}")
        messages = [m.model_dump() for m in chat_request.messages]
        if chat_request.uploaded_filenames:
            messages += [{
                "role": "user",
                "content": ", ".join(chat_request.uploaded_filenames),
                "attachments": [{
                    "file_id": self.stored_files[filename].id,
                    "tools": [{"type": "code_interpreter"}]
                } for filename in chat_request.uploaded_filenames],
            }]
        thread = await async_openai_client.beta.threads.create(
            messages=messages, timeout=120)
        self.threads[thread.id] = thread

        assistant = self.assistants[
            openai_settings.SUMMARIZER_NAME if chat_request.uploaded_filenames else openai_settings.ASSISTANT_NAME]
        async with async_openai_client.beta.threads.runs.stream(
            assistant_id=assistant.id,
            thread_id=thread.id,
            event_handler=ThreadEventHandler(assistant, thread, conversation_id, dt_created, chat_request, ws_sender),
        ) as stream:
            try:
                await stream.until_done()
                logger.info("stream.until_done() completed")
            except Exception as e:
                logger.error(f"stream.until_done() failed: {format_exc()}")

    @measure_exec_seconds(use_logging=True, use_prometheus=True)
    async def no_loose_files(self):
        deletion_tasks = []
        while self.stored_files:
            _, stored_file = self.stored_files.popitem()
            deletion_tasks.append(asyncio.create_task(async_openai_client.files.delete(file_id=stored_file.id)))
        await asyncio.gather(*deletion_tasks)

    @measure_exec_seconds(use_logging=True, use_prometheus=True)
    async def no_loose_threads(self):
        deletion_tasks = []
        while self.threads:
            thread_id, _ = self.threads.popitem()
            deletion_tasks.append(asyncio.create_task(async_openai_client.beta.threads.delete(thread_id=thread_id)))
        await asyncio.gather(*deletion_tasks)

    @measure_exec_seconds(use_logging=True, use_prometheus=True)
    async def refresh_assistants(self):
        assistant_list = await async_openai_client.beta.assistants.list(timeout=openai_settings.HTTPX_TIMEOUT)
        for asst in assistant_list.data:
            self.assistants[asst.name] = asst
        await asyncio.gather(
            self.create_or_update_assistant(
                openai_settings.ASSISTANT_NAME,
                instructions=openai_settings.ASSISTANT_INSTRUCTION,
                model=(
                    openai_settings.REASONING_MODEL
                    if openai_settings.REASONING_MODEL else openai_settings.CHAT_MODEL),
                reasoning_effort=(
                    "medium"
                    if openai_settings.REASONING_MODEL else None),
                timeout=openai_settings.HTTPX_TIMEOUT,
                tools=[],
            ),
            self.create_or_update_assistant(
                openai_settings.SUMMARIZER_NAME,
                instructions=openai_settings.SUMMARIZER_INSTRUCTION,
                model=openai_settings.SUMMARY_MODEL,
                timeout=openai_settings.HTTPX_TIMEOUT,
                tools=[
                    {"type": "code_interpreter"},
                ],
            ),
        )

    @measure_exec_seconds(use_logging=True, use_prometheus=True)
    async def refresh_files(self):
        file_list = await async_openai_client.files.list(
            purpose="assistants",
            timeout=openai_settings.HTTPX_TIMEOUT)
        for stored_file in file_list.data:
            self.stored_files[stored_file.filename] = stored_file
        logger.info(f"found files: {self.stored_files}")


class ThreadEventHandler(AsyncAssistantEventHandler):
    def __init__(self, assistant, thread, conversation_id: int, dt_created: datetime,
                 chat_request: ChatRequest, ws_sender: WebSocketSender):
        self.assistant = assistant
        self.chat_request = chat_request
        self.conversation_id = conversation_id
        self.dt_created = dt_created
        self.thread = thread
        self.ws_sender = ws_sender
        super().__init__()

    @override
    async def on_end(self) -> None:
        logger.info(f"on_end called")
        await self.ws_sender.send_reply(
            self.dt_created, ChatResponse(complete=True, content=f"Thread {self.thread.id} ended.")
        )
        await super().on_end()

    @override
    async def on_message_done(self, message) -> None:
        await self.ws_sender.send_reply(
            self.dt_created,
            ChatResponse(complete=False, content=message.content[0].text.value, model=self.assistant.model)
        )
        await super().on_message_done(message)

    @override
    async def on_tool_call_created(self, tool_call):
        await self.ws_sender.send_reply(
            self.dt_created,
            ChatResponse(complete=False, content=f"One moment, using tool: `{tool_call.type}`.")
        )
        await super().on_tool_call_created(tool_call)
