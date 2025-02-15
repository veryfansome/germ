from openai.lib.streaming import AsyncAssistantEventHandler
from traceback import format_exc
from typing_extensions import override
import asyncio
import logging.config

from bot.api.models import ChatRequest, ChatResponse
from bot.chat import async_openai_client
from bot.websocket import WebSocketSender
from observability.annotations import measure_exec_seconds
from settings import openai_settings

logger = logging.getLogger(__name__)


class AssistantHelper:
    def __init__(self):
        self.assistants = {}
        self.stored_files = {}
        self.threads = {}

    @measure_exec_seconds(use_logging=True, use_prometheus=True)
    async def handle_in_a_thread(self, chat_session_id: int, chat_request_received_id: int,
                                 chat_request: ChatRequest, ws_sender: WebSocketSender):
        if chat_request.uploaded_filenames:
            for filename in chat_request.uploaded_filenames:
                if filename not in self.stored_files:
                    new_stored_file = await async_openai_client.files.create(
                        file=open(f"/tmp/{filename}", "rb"),
                        purpose='assistants',
                        timeout=openai_settings.HTTPX_TIMEOUT,
                    )
                    self.stored_files[filename] = new_stored_file
                    logger.info(f"uploaded: {new_stored_file}")
            thread = await async_openai_client.beta.threads.create(
                messages=[m.model_dump() for m in chat_request.messages] + [{
                    "role": "user",
                    "content": ", ".join(chat_request.uploaded_filenames),
                    "attachments": [{
                        "file_id": self.stored_files[filename].id,
                        "tools": [{"type": "code_interpreter"}]
                    } for filename in chat_request.uploaded_filenames]
                }],
            )
            self.threads[thread.id] = thread

            assistant = self.assistants[openai_settings.ASSISTANT_NAME]
            async with async_openai_client.beta.threads.runs.stream(
                assistant_id=assistant.id,
                thread_id=thread.id,
                event_handler=ThreadEventHandler(
                    assistant, chat_session_id, chat_request_received_id,
                    chat_request, ws_sender),
            ) as stream:
                try:
                    await stream.until_done()
                    logger.info("stream.until_done() completed")
                except Exception as e:
                    logger.error(f"stream.until_done() failed: {format_exc()}")

    @measure_exec_seconds(use_logging=True, use_prometheus=True)
    async def no_loose_files(self):
        deletion_tasks = []
        for stored_file in self.stored_files.values():
            deletion_tasks.append(asyncio.create_task(async_openai_client.files.delete(file_id=stored_file.id)))
        await asyncio.gather(*deletion_tasks)

    @measure_exec_seconds(use_logging=True, use_prometheus=True)
    async def no_loose_threads(self):
        deletion_tasks = []
        for thread_id in self.threads:
            deletion_tasks.append(asyncio.create_task(async_openai_client.beta.threads.delete(thread_id=thread_id)))
        await asyncio.gather(*deletion_tasks)

    @measure_exec_seconds(use_logging=True, use_prometheus=True)
    async def refresh_assistants(self):
        assistant_list = await async_openai_client.beta.assistants.list(timeout=openai_settings.HTTPX_TIMEOUT)
        for asst in assistant_list.data:
            self.assistants[asst.name] = asst
        if not assistant_list.data:
            asst = await async_openai_client.beta.assistants.create(
                name=openai_settings.ASSISTANT_NAME,
                instructions="Be helpful. Always read attached files, using tools when appropriate.",
                model=openai_settings.SUMMARY_MODEL,
                timeout=openai_settings.HTTPX_TIMEOUT,
                tools=[
                    {"type": "code_interpreter"},
                ],
            )
            self.assistants[asst.name] = asst
        logger.info(f"initialized {self.assistants}")

    @measure_exec_seconds(use_logging=True, use_prometheus=True)
    async def refresh_files(self):
        file_list = await async_openai_client.files.list(
            purpose="assistants",
            timeout=openai_settings.HTTPX_TIMEOUT)
        for stored_file in file_list.data:
            self.stored_files[stored_file.filename] = stored_file
        logger.info(f"found files: {self.stored_files}")


class ThreadEventHandler(AsyncAssistantEventHandler):
    def __init__(self, assistant, chat_session_id: int, chat_request_received_id: int,
                 chat_request: ChatRequest, ws_sender: WebSocketSender):
        self.assistant = assistant
        self.chat_request = chat_request
        self.chat_request_received_id = chat_request_received_id
        self.chat_session_id = chat_session_id
        self.ws_sender = ws_sender
        super().__init__()

    @override
    async def on_tool_call_created(self, tool_call):
        _ = asyncio.create_task(
            self.ws_sender.return_chat_response(
                self.chat_request_received_id,
                ChatResponse(complete=False,
                             content=f"One moment. I'm using my `{tool_call.type}` tool.",
                             model=self.assistant.model)))
        await super().on_tool_call_created(tool_call)

    @override
    async def on_message_done(self, message) -> None:
        _ = asyncio.create_task(
            self.ws_sender.return_chat_response(
                self.chat_request_received_id,
                ChatResponse(complete=message.status == "completed",
                             content=message.content[0].text.value,
                             model=self.assistant.model)))
        await super().on_message_done(message)
