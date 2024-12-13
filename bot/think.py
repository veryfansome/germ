import asyncio
from abc import ABC, abstractmethod
from typing import Optional

from openai import OpenAI
from starlette.concurrency import run_in_threadpool

from api.models import ChatMessage
from observability.logging import logging, setup_logging
from settings.openai_settings import HTTPX_TIMEOUT

logger = logging.getLogger(__name__)


class ThinkingPair(ABC):
    @abstractmethod
    async def delegate(self, message: str):
        pass


class PairedThinker(ThinkingPair):
    def __init__(self, model="gpt-4o", name="", pair: ThinkingPair = None,
                 system_message: str = "Play devil's advocate."):
        self._pair: Optional[ThinkingPair] = pair
        self.history: list[ChatMessage] = []
        self.model = model
        self.name = name
        self.stop = False
        self.system_message = system_message

    async def delegate(self, message: str):
        if self.stop:
            return

        logger.debug(f"{self.name} received: {message}")
        logger.info(f"{self.name}: {len(self.history)}")
        self.history.append(ChatMessage(role="user", content=message))

        def _do_completion():
            with OpenAI() as client:
                completion = client.chat.completions.create(
                    messages=[
                        {
                            "role": "system",
                            "content": f"""
- Don't be flowery. Be factual, rational, and efficient.
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

        next_message = await run_in_threadpool(_do_completion)
        asyncio.create_task(self._pair.delegate(next_message))

    async def ruminate(self, idea: str, num_ticks: int = 1 + 2):
        await self.delegate(idea)
        while len(self.history) < num_ticks:
            await asyncio.sleep(1)

    def set_pair(self, pair: ThinkingPair):
        self._pair = pair

    def summarize(self):
        with OpenAI() as client:
            completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": """
- Don't be flowery. Be accurate, efficient, and discerning.
- Don't use Markdown. Minimize blank lines and empty space.
""",
                    },
                    *self.history,
                    {
                        "role": "user",
                        "content": f"List key point. Final thoughts on \"{self.history[0].content}\"",
                    },
                ],
                model=self.model,
                n=1, temperature=0,
                timeout=HTTPX_TIMEOUT)
            return completion.choices[0].message.content


if __name__ == '__main__':
    import argparse

    setup_logging()

    parser = argparse.ArgumentParser(description='Think about an idea.')
    parser.add_argument("-i", "--idea", type=str, help='A seed idea', default="I am?")
    args = parser.parse_args()

    t1 = PairedThinker(name="t1")
    t2 = PairedThinker(name="t2", pair=t1)
    t1.set_pair(t2)

    asyncio.run(t1.ruminate(args.idea, num_ticks=4))
    t1.stop = True
    t2.stop = True
    print(f"\n{t1.summarize()}")
