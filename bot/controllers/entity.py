import asyncio

from api.models import ChatRequest
from bot.chat.openai_handlers import UserProfileConsumer
from bot.graph.idea import SentenceMergeEventHandler, idea_graph
from bot.websocket import WebSocketSender
from observability.logging import logging, setup_logging

logger = logging.getLogger(__name__)


class EntityController(SentenceMergeEventHandler, UserProfileConsumer):
    def __init__(self, interval_seconds: int = 30):
        self.interval_seconds = interval_seconds

    async def on_new_user_profile(self, user_profile, chat_session_id: int, chat_request_received_id: id,
                                  chat_request: ChatRequest, ws_sender: WebSocketSender):
        logger.info(f"on_new_user_profile: {user_profile}")

    async def on_periodic_run(self):
        logger.info("on_periodic_run")

    async def on_sentence_merge(self, node_type: str, sentence_id: int, openai_parameters):
        logger.info(f"on_sentence_merge: {node_type}, sentence_id={sentence_id}, {openai_parameters}")


entity_controller = EntityController()


async def main():
    await entity_controller.on_periodic_run()


if __name__ == "__main__":
    setup_logging()
    while True:
        asyncio.run(main())
