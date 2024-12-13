import asyncio
import random
from abc import ABC, abstractmethod
from typing import Optional

from openai import OpenAI
from starlette.concurrency import run_in_threadpool

from api.models import ChatMessage
from observability.logging import logging, setup_logging
from settings.openai_settings import HTTPX_TIMEOUT

logger = logging.getLogger(__name__)


class DelegatingThinker(ABC):
    @abstractmethod
    async def delegate(self, message: str,
                       initial_call: bool = False,  # Is initial delegate call for an idea or statement.
                       delegate_ok: bool = True):  # Skip delegation if False.
        pass

    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def stop(self):
        pass

    @abstractmethod
    def summarize(self) -> str:
        pass


class PairedThinker(DelegatingThinker):
    def __init__(self, model="gpt-4o-mini", name: str = "", pair: DelegatingThinker = None,
                 system_message: str = "Be thoughtful."):
        self._name = name
        self._pair: Optional[DelegatingThinker] = pair
        self._stop = False
        self.history: list[ChatMessage] = []
        self.model = model
        self.system_message = system_message

    def combined_summary(self) -> str:
        return f"""
{'-' * 100}
## {self._name}
{self.summarize()}
{'-' * 100}
## {self._pair.name()}
{self._pair.summarize()}
{'-' * 100}
"""

    async def delegate(self, message: str,
                       initial_call: bool = False,
                       delegate_ok: bool = True):
        if self._stop:
            return
        elif self._pair is None:
            logger.warning("PairedThinker.delegate() invoked with a `None` pair")

        logger.info(f"{self._name}[hist:{len(self.history)}] received: {message}")

        initial_call_task = None
        if initial_call and self._pair is not None:
            # Let pair have a go too so both see the initial message.
            # Don't delegate because we'll delegate later, and it would create two loops.
            initial_call_task = asyncio.create_task(self._pair.delegate(message, delegate_ok=False))

        self.history.append(ChatMessage(role="user", content=message))

        next_message = await run_in_threadpool(self.do_completion)
        if delegate_ok:
            if initial_call_task is not None:
                await initial_call_task
            if not self._stop and self._pair is not None:
                delegate_task = asyncio.create_task(self._pair.delegate(next_message))

    def do_completion(self):
        with OpenAI() as client:
            completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": f"""
- Be efficient. Use a single concise sentence.
- {self.system_message}
""",
                    },
                    *self.history,
                ],
                model=self.model,
                n=1, temperature=1,
                timeout=HTTPX_TIMEOUT)
            self.history.append(ChatMessage(role="assistant", content=completion.choices[0].message.content))
            return completion.choices[0].message.content

    def name(self) -> str:
        return self._name

    def pair_name(self) -> str:
        return self._pair.name()

    def pair_summarize(self) -> str:
        return self._pair.summarize()

    def reset(self):
        self.history = []
        self._pair.reset()

    async def ruminate(self, idea: str,
                       num_ticks: int = 3):  # Odds end on Self, evens end on Pair
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
        self._pair.stop()

    async def ruminate_randomly(self, idea: str,
                                num_ticks: int = 3):  # Odds end on Self, evens end on Pair
        thinker = random.choice([self, self._pair])
        await thinker.ruminate(idea, num_ticks)
        return thinker

    def set_pair(self, pair: DelegatingThinker):
        self._pair = pair

    def stop(self):
        self._stop = True

    def summarize(self) -> str:
        logger.info(f"{self._name}[hist:{len(self.history)}] generating summary")
        with OpenAI() as client:
            completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": f"""
- Be efficient. Use a single concise sentence.
- {self.system_message}
""",
                    },
                    *self.history,
                    {
                        "role": "user",
                        "content": f"Last word on \"{self.history[0].content}\"?",
                    },
                ],
                model=self.model,
                n=1, temperature=0,
                timeout=HTTPX_TIMEOUT)
            return completion.choices[0].message.content


if __name__ == '__main__':
    import argparse
    import copy
    import itertools

    setup_logging(global_level="INFO")

    # -f, --flavor command args
    ALTRUIST = "altruist"
    CYNIC = "cynic"
    DEVILS_ADVOCATE = "devils_advocate"
    OPEN_MIND = "open_mind"
    PRAGMATIST = "pragmatist"
    SELF_INTEREST = "self_interest"
    UTILITARIAN = "utilitarian"

    FLAVOR_CHOICES = [DEVILS_ADVOCATE, ALTRUIST, OPEN_MIND, SELF_INTEREST]
    THINKER_NAMES = {  # Formated names
        ALTRUIST: "Altruist",
        CYNIC: "Cynic",
        DEVILS_ADVOCATE: "Devil's Advocate",
        OPEN_MIND: "Open mind",
        PRAGMATIST: "Pragmatist",
        SELF_INTEREST: "Self interest",
        UTILITARIAN: "Utilitarian",
    }

    parser = argparse.ArgumentParser(description='Think about an idea.')
    parser.add_argument("-i", "--idea", help='Any idea, simple or complex.',
                        default="People are good.")
    parser.add_argument("-f", "--flavor",
                        choices=FLAVOR_CHOICES, nargs="+", help='Thinker flavors. Pick two. Basic mode ONLY.',
                        default=list(random.sample(FLAVOR_CHOICES, 2)))
    parser.add_argument("--wash", action="store_true", help='Hive mode',
                        default=False)
    args = parser.parse_args()

    thinker_menu = {
        ALTRUIST: PairedThinker(name=THINKER_NAMES[ALTRUIST],
                                system_message="Be altruistic. Champion the needs of others."),
        CYNIC: PairedThinker(name=THINKER_NAMES[CYNIC],
                             system_message="Be cynical. Question motives and ideals."),
        DEVILS_ADVOCATE: PairedThinker(name=THINKER_NAMES[DEVILS_ADVOCATE],
                                       system_message="Be challenging. Play devil's advocate."),
        OPEN_MIND: PairedThinker(name=THINKER_NAMES[OPEN_MIND],
                                 system_message="Be open minded. Run with every idea."),
        PRAGMATIST: PairedThinker(name=THINKER_NAMES[PRAGMATIST],
                                  system_message="Be pragmatic. Focus on what's practical."),
        SELF_INTEREST: PairedThinker(name=THINKER_NAMES[SELF_INTEREST],
                                     system_message="Be greedy. Champion self interest."),
        UTILITARIAN: PairedThinker(name=THINKER_NAMES[UTILITARIAN],
                                   system_message="Be utilitarian. Champion the greatest good."),
    }

    async def basic():
        summaries = []
        thinkers = []

        if DEVILS_ADVOCATE in args.flavor:
            thinkers.append(thinker_menu[DEVILS_ADVOCATE])
        if ALTRUIST in args.flavor:
            thinkers.append(thinker_menu[ALTRUIST])
        if OPEN_MIND in args.flavor:
            thinkers.append(thinker_menu[OPEN_MIND])
        if SELF_INTEREST in args.flavor:
            thinkers.append(thinker_menu[SELF_INTEREST])

        # Two max
        thinker_len = len(thinkers)
        if thinker_len == 1:
            thinkers += list(random.sample({n: t for n, t in thinker_menu.items() if n != args.thinker[0]}, 1))
        elif thinker_len > 2:
            thinkers = list(random.sample(thinkers, 2))

        thinkers[0].set_pair(thinkers[1])
        thinkers[1].set_pair(thinkers[0])
        thinker = await thinkers[0].ruminate_randomly(args.idea)
        summaries.append(thinker.combined_summary())
        print(f"\n# Idea: {args.idea}")
        print(*summaries)

    async def wash(limiter: asyncio.Semaphore):
        summaries = {k: [] for k in THINKER_NAMES.values()}
        lead_thinkers: [PairedThinker] = []
        # Do all unique combos
        for p1, p2 in list(itertools.combinations(thinker_menu.values(), 2)):
            p1_copy = copy.deepcopy(p1)
            p2_copy = copy.deepcopy(p2)
            p1_copy.set_pair(p2_copy)
            p2_copy.set_pair(p1_copy)
            lead_thinkers.append(p1_copy)

        async def _limited_task(lead: PairedThinker):
            # Throttle how many we do at once
            async with limiter:
                logger.info(f"Starting {lead.name()} vs {lead.pair_name()}")
                return await lead.ruminate_randomly(args.idea)

        wash_tasks = []
        for thinker in lead_thinkers:
            wash_tasks.append(_limited_task(thinker))
        lead_thinkers = await asyncio.gather(*wash_tasks)
        for thinker in lead_thinkers:
            summaries[thinker.name()].append(thinker.summarize())
            summaries[thinker.pair_name()].append(thinker.pair_summarize())
        for thinker_name in summaries.keys():
            print(f"## {thinker_name}\n")
            print(''.join([f'- {s}\n' for s in summaries[thinker_name]]))

    if args.wash:
        semaphore = asyncio.Semaphore(2)
        asyncio.run(wash(semaphore))
    else:
        asyncio.run(basic())
