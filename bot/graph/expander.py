import asyncio
import json
import copy
from starlette.concurrency import run_in_threadpool

from bot.graph.idea import get_idea_graph, IdeaGraph
from bot.think.pair import PairedThinker
from bot.think.single import SingleThreadedMultiFlavorThinker
from observability.logging import logging, setup_logging

logger = logging.getLogger(__name__)
idea_graph: IdeaGraph = get_idea_graph(__name__)




async def foo(idea: str):
    #thinker = SingleThreadedMultiFlavorThinker()
    #thinker.add_first_message(idea)
    #for i in range(3):
    #    thinker.round_table()

    # Brainstorm:
    # - Based on context, infer something what was said that must be true.
    # - keep inferring until you run out of new differentiated ideas
    # - Rinse/Repeat from different parts of the idea-space?
    # - pull any two randon ideas and consider them together, then make an inference that must be true.
    # - I usually use a low temperature but maybe for this kind of thing, the temperature should be up so we take a bigger leap so to speak
    # - look into frequency and other parameters

    # - Time needs to have a retention policy. It needs to simulate the perception of an instance in time, rather than a continous river
    # - Once a time node is detached, the thing attached to it can be attached to a new date entity or date node
    # - Maybe there needs to be Hour, Day, Week, Month, all with retention policies where things roll off from one tier to the next.

    p1 = PairedThinker(name="thinker 1", system_message=(
        "You are a rational and lateral thinker. "
        "Assume what the user says is always inviolably true. "
        "Use a declarative statement to infer something else that must also be true."
    ))
    p2 = copy.deepcopy(p1)
    p1.set_pair(p2)
    p2.set_pair(p1)
    await p1.ruminate(idea)
    logger.info(json.dumps({"thoughts": p1.summarize()}, indent=4))


async def main():
    idea_results = idea_graph.get_random_ideas(count=1)
    candidate = idea_results[0]["idea"]["text"]
    logger.info(f"expander candidate: {candidate}")
    await foo(candidate)


if __name__ == "__main__":
    setup_logging()
    asyncio.run(main())
