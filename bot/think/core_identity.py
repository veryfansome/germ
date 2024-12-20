import asyncio
from starlette.concurrency import run_in_threadpool

from bot.graph.idea import get_idea_graph, IdeaGraph
from bot.lang.examples import sentences as sentence_examples
from observability.logging import logging

logger = logging.getLogger(__name__)
idea_graph: IdeaGraph = get_idea_graph(__name__)


async def main():
    """
    Reinforce core identity concepts by creating connections between these ideas and the current Time node.

    :return:
    """
    logger.info("Reinforcing core identity")
    tasks = []
    for sentence in sentence_examples.core_identity:
        tasks.append(run_in_threadpool(idea_graph.add_sentence, sentence,
                                       flair_features=None, openai_features=None))
        tasks.append(run_in_threadpool(idea_graph.add_idea, sentence))
    await asyncio.gather(*tasks)
