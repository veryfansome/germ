import asyncio
from starlette.concurrency import run_in_threadpool

from bot.graph.idea import get_idea_graph, IdeaGraph
from bot.lang.examples import sentences as sentence_examples
from observability.logging import logging, setup_logging

logger = logging.getLogger(__name__)
idea_graph: IdeaGraph = get_idea_graph(__name__)


async def main():
    """
    Reinforce core identity concepts by creating connections between these ideas and the current Time node.

    :return:
    """
    logger.info("Reinforcing core identity")
    tasks = []
    current_rounded_time, _, _ = await idea_graph.add_time()
    for sentence in sentence_examples.core_identity:
        tasks.append(idea_graph.add_sentence(
            sentence, current_rounded_time=current_rounded_time,
            flair_features=None, openai_features=None, openai_sentence_type=None))
    await asyncio.gather(*tasks)


if __name__ == '__main__':
    setup_logging()
    asyncio.run(main())
