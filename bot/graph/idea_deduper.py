import asyncio
import json
from openai import OpenAI
from starlette.concurrency import run_in_threadpool

from bot.graph.idea import get_idea_graph, IdeaGraph
from observability.logging import logging, setup_logging

logger = logging.getLogger(__name__)
idea_graph: IdeaGraph = get_idea_graph(__name__)

tool_func_name = "merge_duplicates"
tool = {
    "type": "function",
    "function": {
        "name": tool_func_name,
        "description": "Merge ideas that are the same.",
        "parameters": {
            "type": "object",
            "properties": {
                "is_duplicate": {
                    "type": "string",
                    "description": "true, if the statements convey the same idea without significant nuance.",
                    "enum": ["true", "false"]
                },
            },
            "required": ["is_duplicate"],
            "additionalProperties": False,
        },
    }
}


def find_duplicates(ideas: list[tuple[str, str]]):
    logger.info(f"Connection candidates: {ideas}")
    for idea_pair in ideas:
        with OpenAI() as client:
            completion = client.chat.completions.create(
                model="gpt-4o-mini", n=1, temperature=0, timeout=60,
                messages=[
                    {
                        "role": "system",
                        "content": "Consider the statements in the list. "
                                   "Do they convey the same fundamental idea? "
                    },
                    {
                        "role": "user",
                        "content": json.dumps(list(idea_pair), indent=4),
                    },
                ],
                tool_choice={
                    "type": "function",
                    "function": {"name": tool_func_name}
                },
                tools=[tool])
            is_duplicate = json.loads(completion.choices[0].message.tool_calls[0].function.arguments)["is_duplicate"]
            logger.info(f"is_duplicate {is_duplicate}")
            if is_duplicate == "true":
                # If two sentences convey the same idea, keep the one that uses fewer words.
                if len(idea_pair[0].split()) < len(idea_pair[1].split()):
                    idea_graph.merge_ideas(idea_pair[0], idea_pair[1])
                else:
                    idea_graph.merge_ideas(idea_pair[1], idea_pair[0])


async def main():
    topic_results, idea_results = idea_graph.get_similar_but_disconnected_ideas_by_random_topic()
    candidate_tuples = [(record["i1"]["text"], record["i2"]["text"]) for record in idea_results]
    await run_in_threadpool(find_duplicates, candidate_tuples)


if __name__ == "__main__":
    setup_logging()
    asyncio.run(main())
