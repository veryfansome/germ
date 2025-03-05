import asyncio

from bot.chat import async_openai_client, openai_beta


async def do_chat_completion():
    scenario = {
        # TODO:
        #  - Leverage emotions experienced with user.
        #  - Prioritize based on positive responses from the user.
        #  - Leverage information learned about the user.
        #  - Query the graph.
        "You know who you're talking to.": False,  # Or name
        "You know what time it is.": False,  # Or time
        "You know where you are.": False,  # Or location
        # ^ doesn't lead model to ask for these things - needs more playing with
    }
    completion = await async_openai_client.chat.completions.create(
        max_completion_tokens=30,
        messages=[
            {"role": "system",
             "content": "Use the provided scenario info. Role-play someone like Han Solo but in the real world."},
            {"role": "user",
             "content": str(scenario)},
            {"role": "user",
             "content": "You say:"}
        ],
        model="gpt-4o",
        n=1, timeout=60)
    print(completion.choices[0].message.content)
    return completion


if __name__ == '__main__':
    asyncio.run(do_chat_completion())