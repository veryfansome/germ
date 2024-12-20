from starlette.concurrency import run_in_threadpool

from bot.graph.idea import get_idea_graph
from observability.logging import logging

logger = logging.getLogger(__name__)
idea_graph = get_idea_graph(__name__)

# As the bot encounters texts, either through user chats, news articles, or any other document, it will create Sentence
# nodes in its graph.

# Periodically, it should run various workflows to ruminate sentences to distill them into Idea nodes.


async def main():
    logger.info("Distilling new ideas")
