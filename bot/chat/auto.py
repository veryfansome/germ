import logging
from openai import OpenAI

from api.models import ChatMessage

logger = logging.getLogger(__name__)


class SingleSentenceChatter:
    def __init__(self, completion_model: str = "gpt-4o-mini",
                 history: list[ChatMessage] = None,
                 name: str = "",
                 summarization_model: str = "gpt-4o-mini",
                 system_message: str = None):
        self.completion_model = completion_model
        self.history = history if history is not None else []
        self.name = name
        self.summarization_model = summarization_model
        self.system_message = system_message

    def do_completion(self):
        with OpenAI() as client:
            completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": f"""
- Deliver the point only. Use a single concise sentence with simple structure.
- {self.system_message}
""",
                    },
                    *self.history,
                ],
                model=self.completion_model,
                n=1, temperature=1,
                timeout=30)
            completion_content = _u2019_replace(completion.choices[0].message.content)
            self.history.append(ChatMessage(role="assistant", content=completion_content))
            return completion_content

    def summarize(self) -> str:
        logger.info(f"{self.name}[hist:{len(self.history)}] generating summary")
        with OpenAI() as client:
            completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": f"""
- Deliver the point only. Use a single concise sentence with simple structure.
- {self.system_message}
""",
                    },
                    *self.history,
                    {
                        "role": "user",
                        "content": f"Update your take on \"{self.history[0].content}\"",
                    },
                ],
                model=self.summarization_model,
                n=1, temperature=1,
                timeout=60)
            return _u2019_replace(completion.choices[0].message.content)


def _u2019_replace(text: str) -> str:
    return text.replace("\\u2019", "'")  # This encoding issue surfaces in generated text
