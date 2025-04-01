from starlette.concurrency import run_in_threadpool
import asyncio
import logging

from bot.api.models import ChatRequest, ChatResponse
from bot.graph.control_plane import ControlPlane
from bot.lang.parsers import extract_markdown_page_elements
from bot.websocket import (WebSocketReceiveEventHandler, WebSocketSendEventHandler, WebSocketSender,
                           WebSocketSessionMonitor)
from observability.annotations import measure_exec_seconds

logger = logging.getLogger(__name__)


class ChatController(WebSocketReceiveEventHandler, WebSocketSendEventHandler, WebSocketSessionMonitor):

    def __init__(self, control_plane: ControlPlane, remote: WebSocketReceiveEventHandler):
        self.control_plane = control_plane
        self.node_types = {
            "block_code": "CodeBlock",
            "list": "Paragraph",
            "paragraph": "Paragraph",
        }
        self.remote = remote

    @measure_exec_seconds(use_logging=True, use_prometheus=True)
    async def on_receive(self, chat_session_id: int, chat_request_received_id: int, chat_request: ChatRequest,
                         ws_sender: WebSocketSender):
        remote_response_task = asyncio.create_task(
            self.remote.on_receive(chat_session_id, chat_request_received_id, chat_request, ws_sender),
        )
        # TODO:
        #  - implement local completions
        #  - maybe even user profiling should be moved here...
        elements = await run_in_threadpool(extract_markdown_page_elements, chat_request.messages[-1].content)
        await self.process_markdown_element(
            elements,
            self.control_plane.link_page_element_to_chat_request, [chat_request_received_id],
            {
                "_": {"deferred_labeling": False},
                "chat_request_received_id": chat_request_received_id,
                "chat_session_id": chat_session_id,
            })
        await remote_response_task

    async def on_send(self,
                      chat_response_sent_id: int,
                      chat_response: ChatResponse,
                      chat_session_id: int,
                      chat_request_received_id: int = None):
        elements = await run_in_threadpool(extract_markdown_page_elements, chat_response.content)
        await self.process_markdown_element(
            elements,
            self.control_plane.link_page_element_to_chat_response, [chat_response_sent_id],
            {
                "chat_request_received_id": chat_request_received_id,
                "chat_response_sent_id": chat_response_sent_id,
                "chat_session_id": chat_session_id,
            })

    async def on_tick(self, chat_session_id: int, ws_sender: WebSocketSender):
        logger.info(f"chat session {chat_session_id} is still active")

    async def process_markdown_element(self, markdown_elements,
                                       element_link_func, element_link_func_args,
                                       session_attrs):
        async_tasks = []
        h1 = None
        h2 = None
        h3 = None
        h4 = None
        h5 = None
        h6 = None
        last_element_type = None
        last_element_attrs = None
        list_elements = []
        list_type = None
        for element in markdown_elements:
            logger.debug(f"markdown element: {element}")
            merged_element_attrs = {
                **session_attrs,
                "h1": str(h1),
                "h2": str(h2),
                "h3": str(h3),
                "h4": str(h4),
                "h5": str(h5),
                "h6": str(h6),
                "list_type": str(list_type),
            }

            if element[0] == "block_text":
                # NOTE: block_text duplicates list_item
                continue
            elif element[0] == "list_item":
                list_elements.append(element[1])
                continue
            elif element[0] == "list":
                list_type = "ul" if element[1] is False else "ol"
                # Store as paragraph
                _, text_block_id, paragraph_tasks = await self.control_plane.add_paragraph(
                    "\n".join([(f"• {e}" if list_type == 'ul' else f"{i+1}. {e}") for i, e in enumerate(list_elements)]),
                    {**merged_element_attrs, "list_size": len(list_elements), "list_type": list_type})
                async_tasks.extend(paragraph_tasks)
                this_element_attrs = {"text_block_id": text_block_id}
                list_elements = []
                list_type = None
            else:
                list_elements = []
                list_type = None

                # `paragraph` and `code_block` are ordered at the top for frequency
                if element[0] == "paragraph":
                    _, text_block_id, paragraph_tasks = await self.control_plane.add_paragraph(
                        element[1], merged_element_attrs)
                    async_tasks.extend(paragraph_tasks)
                    this_element_attrs = {"text_block_id": text_block_id}
                elif element[0] == "block_code":
                    _, text_block_id, code_block_tasks = await self.control_plane.add_code_block(
                        element[2], {"language": str(element[1]), **merged_element_attrs})
                    async_tasks.extend(code_block_tasks)
                    this_element_attrs = {"text_block_id": text_block_id}
                elif element[0] == "heading":
                    if element[1] == 1:
                        h1 = element[2]
                        h2 = None
                        h3 = None
                        h4 = None
                        h5 = None
                        h6 = None
                    elif element[1] == 2:
                        h2 = element[2]
                        h3 = None
                        h4 = None
                        h5 = None
                        h6 = None
                    elif element[1] == 3:
                        h3 = element[2]
                        h4 = None
                        h5 = None
                        h6 = None
                    elif element[1] == 4:
                        h4 = element[2]
                        h5 = None
                        h6 = None
                    elif element[1] == 5:
                        h5 = element[2]
                        h6 = None
                    elif element[1] == 6:
                        h6 = element[2]
                    this_element_attrs = None
                else:
                    logger.info(f"unsupported element type: {element}")
                    continue

            if this_element_attrs:
                async_tasks.append(
                    asyncio.create_task(element_link_func(
                        self.node_types[element[0]], this_element_attrs, *element_link_func_args))
                )
                if last_element_type is not None and last_element_attrs is not None:
                    async_tasks.append(
                        asyncio.create_task(self.control_plane.link_successive_page_elements(
                            last_element_type, last_element_attrs,
                            self.node_types[element[0]], this_element_attrs,
                            session_attrs))
                    )

                last_element_attrs = this_element_attrs
                last_element_type = self.node_types[element[0]]
        await asyncio.gather(*async_tasks)
