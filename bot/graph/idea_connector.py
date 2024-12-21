import asyncio
import json
from openai import OpenAI
from starlette.concurrency import run_in_threadpool

from bot.graph.idea import get_idea_graph, IdeaGraph
from observability.logging import logging, setup_logging

logger = logging.getLogger(__name__)
idea_graph: IdeaGraph = get_idea_graph(__name__)

tool_func_name = "report_connections"
tool = {
    "type": "function",
    "function": {
        "name": tool_func_name,
        "description": "Report connects discovered between ideas.",
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


def find_connection(ideas):
    logger.info(f"Connection candidates: {ideas}")
    #with OpenAI() as client:
    #    completion = client.chat.completions.create(
    #        model="gpt-4o-mini", n=1, temperature=0, timeout=60,
    #        messages=[
    #            {
    #                "role": "system",
    #                "content": "Consider the statements in the list. "
    #                           "Can all statements be true at the same time? "
    #            },
    #            {
    #                "role": "user",
    #                "content": json.dumps(ideas, indent=4),
    #            },
    #        ],
    #        tool_choice={
    #            "type": "function",
    #            "function": {"name": tool_func_name}
    #        },
    #        tools=[tool])
    #    tool_params = completion.choices[0].message.tool_calls[0].function.arguments
    #    logger.info(tool_params)


async def main():
    topic_results, idea_results = idea_graph.get_similar_but_disconnected_ideas_by_random_topic()
    candidate_tuples = [(record["i1"]["text"], record["i2"]["text"]) for record in idea_results]
    await run_in_threadpool(find_connection, candidate_tuples)


if __name__ == "__main__":
    setup_logging()
    asyncio.run(main())

    #find_inconsistency(["I love cats.", "I hate animals."])
    #find_inconsistency(["I want to live a long life.", "I can keep smoking, I won't get cancer."])
    #find_inconsistency(["John lied to me.", "John is a reliable friend."])
    #find_inconsistency(["I respect the rule of law.", "If I can get away with it, I should cheat to get ahead."])
    #find_inconsistency(["We should reduce dependence on fossil fuels.", "Drill baby, drill!"])
    #find_inconsistency(["We should reduce dependence on fossil fuels.", "Buy oil stocks when crude prices are lows."])

