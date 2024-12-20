import asyncio
import copy
import itertools
import json
import random
from abc import ABC, abstractmethod
from typing import Optional

from openai import OpenAI
from starlette.concurrency import run_in_threadpool

from api.models import ChatMessage
from bot.think.flavors import THINKER_FLAVORS
from bot.chat.auto import SingleSentenceChatter
from observability.logging import logging, setup_logging

logger = logging.getLogger(__name__)


class DelegatingThinker(SingleSentenceChatter, ABC):
    @abstractmethod
    async def delegate(self, message: str,
                       initial_call: bool = False,  # Is initial delegate call.
                       delegate_ok: bool = True):  # Skip delegation if False.
        pass

    @abstractmethod
    def stop(self):
        pass


class PairedThinker(DelegatingThinker):
    def __init__(self, name: str = "", pair: DelegatingThinker = None,
                 history: list[ChatMessage] = None, system_message: str = "Be thoughtful."):
        super().__init__(history=history if history is not None else [],
                         name=name,
                         system_message=system_message)
        self._stop = False
        self.pair: Optional[DelegatingThinker] = pair
        self.summarization_model = "gpt-4o-mini"

    async def delegate(self, message: str,
                       initial_call: bool = False,
                       delegate_ok: bool = True):
        if self._stop:
            return
        elif self.pair is None:
            logger.warning("PairedThinker.delegate() invoked with a `None` pair")

        logger.info(f"{self.name}[hist:{len(self.history)}] received: {message}")

        initial_call_task = None
        if initial_call and not self._stop and self.pair is not None:
            # Let pair have a go too so both see the initial message.
            # Don't delegate because we'll delegate later, and it would create two loops.
            initial_call_task = asyncio.create_task(self.pair.delegate(message, delegate_ok=False))

        self.history.append(ChatMessage(role="user", content=message))
        if not self._stop:
            next_message = await run_in_threadpool(self.do_completion)
            if delegate_ok:
                if initial_call_task is not None:
                    await initial_call_task
                if not self._stop and self.pair is not None:  # Check stop in case we waited
                    delegate_task = asyncio.create_task(self.pair.delegate(next_message))

    async def ruminate(self, idea: str,
                       num_ticks: int = 3):  # Odds end on Self, evens end on Pair
        """
        Conduct a dialogue between two PairedThinkers.

        :param idea:
        :param num_ticks:
        :return:
        """
        # ticks   1       2       3
        # idea -> Self -> Pair -> Self
        # idea -> Pair
        # Self's history should have:
        # - Initial idea
        # - Initial Self's response send to Pair
        # - Pair's response on Self's response
        # - Self's response on Pair's response
        # Pair's history should have:
        # - Initial idea
        # - Initial Pair's response
        # - Initial Self's response
        # - Pair's response on Self's response
        # - Self's response on Pair's response
        await self.delegate(idea, initial_call=True)
        stop_len = (num_ticks + 1)  # Plus 1 for the first message
        while len(self.history) < stop_len:
            await asyncio.sleep(1)
        self.stop()
        self.pair.stop()

    async def ruminate_randomly(self, idea: str,
                                num_ticks: int = 3):  # Odds end on Self, evens end on Pair
        thinker = random.choice([self, self.pair])
        await thinker.ruminate(idea, num_ticks)
        return thinker

    def set_pair(self, pair: DelegatingThinker):
        self.pair = pair

    def stop(self):
        self._stop = True


async def cross_ruminate(idea: str,
                         thinkers: dict[str, PairedThinker] = None,
                         limiter: asyncio.Semaphore = asyncio.Semaphore(1)):
    """
    Execute all possible ruminate combinations, with each pair having distinct histories, then distill
    the arguments from the discussions.

    :param idea:
    :param thinkers:
    :param limiter:
    :return:
    """
    thinkers = thinkers if thinkers is not None else {
        n: PairedThinker(name=n, system_message=s) for n, s in THINKER_FLAVORS.items()}

    summaries = {k: [] for k in thinkers.keys()}  # Allows grouping by thinker type
    lead_thinkers: [PairedThinker] = []
    for p1, p2 in list(itertools.combinations(thinkers.values(), 2)):  # All combos
        # Copy or histories will be jumbled
        p1_copy = copy.deepcopy(p1)
        p2_copy = copy.deepcopy(p2)
        p1_copy.set_pair(p2_copy)
        p2_copy.set_pair(p1_copy)
        lead_thinkers.append(p1_copy)

    async def _throttled_rumination(lead: PairedThinker):
        # Throttle how many we do at once
        async with limiter:
            logger.info(f"Starting {lead.name} vs {lead.pair.name}")
            return await lead.ruminate_randomly(idea)

    blobs = []
    tasks = []
    for thinker in lead_thinkers:
        tasks.append(_throttled_rumination(thinker))
    lead_thinkers = await asyncio.gather(*tasks)
    for thinker in lead_thinkers:
        summaries[thinker.name].append(thinker.summarize())
        summaries[thinker.pair.name].append(thinker.pair.summarize())
    # Organized by thinker types
    for thinker_name in summaries.keys():
        blobs.append(f"## Thinker: {thinker_name}\n" + (''.join([f'- {s}\n' for s in summaries[thinker_name]])))
    with OpenAI() as client:
        completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "Retain distinct thoughts. Condense duplicative text.",
                },
                {
                    "role": "user",
                    "content": "# Distill these thoughts\n---\n" + ("\n".join(blobs)),
                },
            ],
            model="gpt-4o-mini",
            n=1, temperature=1,
            timeout=120,
            tool_choice={
                "type": "function",
                "function": {"name": "format_thoughts"}
            },
            tools=[{
                "type": "function",
                "function": {
                    "name": "format_thoughts",
                    "description": "Format thoughts for user presentation.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "thoughts": {
                                "type": "array",
                                "description": "Distilled thoughts.",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "thought": {
                                            "type": "string",
                                            "description": "Distilled thought."
                                        },
                                        "thinkers": {
                                            "type": "array",
                                            "items": {
                                                "type": "string",
                                                "description": "Thinker."
                                            }
                                        }
                                    }
                                }
                            }
                        },
                        "required": ["thoughts"],
                        "additionalProperties": False,
                    },
                }
            }]
        )
        return json.loads(completion.choices[0].message.tool_calls[0].function.arguments)


async def ruminate_randomly(idea: str, thinkers: list[PairedThinker] = None):
    """
    Ruminate once using a randomized thinker pair.

    :param thinkers:
    :param idea:
    :return:
    """
    if thinkers is None:
        thinkers = [PairedThinker(name=flv, system_message=THINKER_FLAVORS[flv])
                    for flv in random.sample(sorted(THINKER_FLAVORS.keys()), 2)]
    thinkers[0].set_pair(thinkers[1])
    thinkers[1].set_pair(thinkers[0])
    thinker = await thinkers[0].ruminate_randomly(idea)
    return {
        "thoughts": [{
            "thought": thinker.summarize(),
            "thinkers": [thinker.name],
        }, {
            "thought": thinker.pair.summarize(),
            "thinkers": [thinker.pair.name],
        }]}


if __name__ == '__main__':
    import argparse

    setup_logging(global_level="INFO")
    parser = argparse.ArgumentParser(description='Think about an idea.')
    parser.add_argument("--cross", action="store_true", help='Cross-ruminate.',
                        default=False)
    parser.add_argument("-f", "--flavor", help='Thinker flavors.',
                        nargs="+", default=[])
    parser.add_argument("-i", "--idea", help='Any idea, simple or complex.',
                        default="People are good.")
    args = parser.parse_args()

    if args.cross:
        args_thinkers = {flv: PairedThinker(name=flv, system_message=THINKER_FLAVORS[flv]) for flv in args.flavor}
        semaphore = asyncio.Semaphore(2)
        print(json.dumps(asyncio.run(cross_ruminate(
            args.idea, limiter=semaphore, thinkers=None if len(args_thinkers) == 0 else args_thinkers)), indent=4))
    else:
        args_thinkers = [PairedThinker(name=flv, system_message=THINKER_FLAVORS[flv]) for flv in args.flavor]
        print(json.dumps(asyncio.run(ruminate_randomly(
            args.idea, thinkers=None if len(args_thinkers) == 0 else args_thinkers)), indent=4))
