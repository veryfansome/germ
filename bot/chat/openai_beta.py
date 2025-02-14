import asyncio
import logging.config

from bot.chat import async_openai_client
from observability.logging import setup_logging

logger = logging.getLogger(__name__)


async def kick_sand():
    #assistant_list = await async_openai_client.beta.assistants.list()
    #logger.info(assistant_list)
    #assistant = await async_openai_client.beta.assistants.create(
    #    name="Document Summarizer",
    #    instructions=("You are a personal Document summarizer."
    #                  "Read documents and summarize them accurately and succinctly."),
    #    tools=[{"type": "code_interpreter"}],
    #    model="gpt-4o",
    #)
    #logger.info(assistant)
    #assistant_list = await async_openai_client.beta.assistants.list()
    #logger.info(assistant_list)
    #asst_3N3jofk5RFC0vriRZBkgW92H

    #example_file = await async_openai_client.files.create(
    #    file=open("/src/bot/static/deepseek.pdf", "rb"),
    #    purpose='assistants'
    #)
    #logger.info(example_file)
    #file_list = await async_openai_client.files.list()
    #logger.info(file_list)

    #thread = await async_openai_client.beta.threads.create(
    #    messages=[
    #        {
    #            "role": "user",
    #            "content": "Read this deepseek.pdf and summarize what it's about.",
    #            "attachments": [
    #                {
    #                    #"file_id": example_file.id,
    #                    "file_id": "file-D4akoz8VetycrmQmcHRNxD",
    #                    "tools": [{"type": "code_interpreter"}]
    #                }
    #            ]
    #        }
    #    ]
    #)
    #logger.info(thread)
    #thread_runs = await async_openai_client.beta.threads.runs.create(
    #    assistant_id="asst_3N3jofk5RFC0vriRZBkgW92H",
    #    thread_id="thread_QSqUeSX1foOj5byjLwdEf27W"
    #)
    #logger.info(thread_runs)
    thread_message_list = await async_openai_client.beta.threads.messages.list(
        thread_id="thread_QSqUeSX1foOj5byjLwdEf27W")
    logger.info(thread_message_list)


if __name__ == "__main__":
    setup_logging()
    asyncio.run(kick_sand())
