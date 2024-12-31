import asyncio
import json
from openai import OpenAI
from starlette.concurrency import run_in_threadpool

from bot.graph.idea import idea_graph
from observability.logging import logging, setup_logging

logger = logging.getLogger(__name__)

tool_func_name = "report_inconsistency"
tool = {
    "type": "function",
    "function": {
        "name": tool_func_name,
        "description": "Report logical inconsistencies.",
        "parameters": {
            "type": "object",
            "properties": {
                "inconsistency": {
                    "type": "string",
                    "description": "true, if all statements cannot be simultaneously true, else false.",
                    "enum": ["true", "false"]
                },
            },
            "required": ["inconsistency"],
            "additionalProperties": False,
        },
    }
}


def find_inconsistency(ideas: list[str]):
    logger.info(f"Inconsistencies candidates: {ideas}")
    with OpenAI() as client:
        completion = client.chat.completions.create(
            model="gpt-4o-mini", n=1, temperature=0, timeout=60,
            messages=[
                {
                    "role": "system",
                    "content": "Consider the statements in the list. "
                               "Can all statements be true at the same time? "
                },
                {
                    "role": "user",
                    "content": json.dumps(ideas, indent=4),
                },
            ],
            tool_choice={
                "type": "function",
                "function": {"name": tool_func_name}
            },
            tools=[tool])
        tool_params = completion.choices[0].message.tool_calls[0].function.arguments
        logger.info(tool_params)


async def main():
    idea_records = idea_graph.get_random_ideas(2)
    idea_texts = [record["idea"]["text"] for record in idea_records]
    await run_in_threadpool(find_inconsistency, idea_texts)


if __name__ == "__main__":
    setup_logging()
    while True:
        asyncio.run(main())

    # find_inconsistency(["I love cats.", "I hate animals."])
    # find_inconsistency(["I want to live a long life.", "I can keep smoking, I won't get cancer."])
    # find_inconsistency(["John lied to me.", "John is a reliable friend."])
    # find_inconsistency(["I respect the rule of law.", "If I can get away with it, I should cheat to get ahead."])
    # find_inconsistency(["We should reduce dependence on fossil fuels.", "Drill baby, drill!"])
    # find_inconsistency(["We should reduce dependence on fossil fuels.", "Buy oil stocks when crude prices are lows."])

    # TODO: Just looking at statements side by side using a LLM is not enough because the LLM doesn't have access
    #       to context from the graph. The inconsistency checker actually needs to query the graph as we would during a
    #       human user interaction - this can be a first test of that.

    # idea_graph.add_sentence("I am a little furry dog.")
    # idea_graph.add_sentence("I am a human man in a funny coat.")
    # idea_graph.add_sentence("I have mother name Loraine.")
