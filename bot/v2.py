from bot.v1 import DEFAULT_CHAT_MODEL, OpenAIChatBotV1
from observability.logging import logging

logger = logging.getLogger(__name__)


class OpenAIChatBotV2(OpenAIChatBotV1):

    def __init__(self, model=DEFAULT_CHAT_MODEL):
        super().__init__(model)
