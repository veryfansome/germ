from openai import OpenAI
import json
import random

from api.models import ChatMessage, ChatRequest
from chat.openai_handlers import messages_to_transcript
from lang.inflect import INFLECT_ENGINE
from ml.bert_classifier import BertClassificationPredictor
from settings.openai_settings import DEFAULT_CHAT_MODEL

_imperatives__create = [
    "craft", "create", "come up with", "generate", "make", "produce", "whip up"
]
_imperatives__mood_words = [
    "can you", "can you please", "could you", "could you please", "kindly",
    "please", "would you please"
]
_imperatives__transitive_verbs = [
    'command', 'compel', 'demand', 'need', 'require', 'want'
]
_prepositions = [
    'about', 'above', 'after', 'against', 'among', 'at', 'before', 'below', 'between', 'by', 'down', 'during',
    'for', 'from', 'in', 'near', 'of', 'off', 'on', 'over', 'since', 'through', 'to', 'under', 'until' 'up',
    'with', 'within', 'without',
]


class ActivationTrainingExample:
    def __init__(self, labels: dict[str, str], messages: list[ChatMessage]):
        self.labels: dict[str, str] = labels
        self.messages: list[ChatMessage] = messages
        self.transcript_text: str = messages_to_transcript(self.to_chat_request())

    def to_chat_request(self) -> ChatRequest:
        return ChatRequest(messages=self.messages)


class ActivationTrainer:
    def __init__(self,
                 examples: list[ActivationTrainingExample],
                 trainees: dict[str, BertClassificationPredictor],
                 rounds: int):
        self.examples: list[ActivationTrainingExample] = examples
        self.trainees: dict[str, BertClassificationPredictor] = trainees
        self.rounds: int = rounds

    def train(self):
        for i in range(self.rounds):
            random.shuffle(self.examples)
            for exp in self.examples:
                for image_model_name in exp.labels.keys():
                    transcript_embeddings = self.trainees[image_model_name].generate_embeddings(exp.transcript_text)
                    self.trainees[image_model_name].train_bert_classifier(
                        exp.labels[image_model_name], transcript_embeddings
                    )
                    self.trainees[image_model_name].save()


def a_word(noun: str) -> str:
    return INFLECT_ENGINE.a(noun).split(' ')[0]


def amount_word(what, amount: int) -> str:
    return a_word(what) if amount == 1 else (random.choice([
        INFLECT_ENGINE.number_to_words(amount),
        *['a bunch of'],
    ]) if amount > 5 else random.choice([
        INFLECT_ENGINE.number_to_words(amount),
        *['a few'],
    ]))


def new_create_imperative(what: str, amount: int):
    what = what if amount == 1 else INFLECT_ENGINE.plural(what)
    new_amount_word = amount_word(what, amount)
    return random_capitalization(''.join((
        random.choice([
            # - Generate
            random.choice(_imperatives__create),
            # - Please generate
            f"{random.choice(_imperatives__mood_words)} {random.choice(_imperatives__create)}",
            # - I want to generate
            f"i {random.choice(_imperatives__transitive_verbs)}{random.choice(['', ' you'])} to {random.choice(_imperatives__create)}",
        ]) + ' ',
        # - generate an image
        f"{new_amount_word} {singular_or_plural(what, amount)}",
    )))


def random_capitalization(blob: str):
    return random.choice([blob, blob.capitalize()])


def sentence_completion_candidates(seed: str, num_to_return: int,
                                   allow_changes: bool = True,
                                   optional_components: list[str] = None,
                                   require_all_of: list[str] = None,
                                   require_one_of: list[str] = None,
                                   ) -> list[str]:
    with OpenAI() as client:
        completion = client.chat.completions.create(
            messages=([{
                "role": "system",
                "content": """
- Generate results on an increasing scale of complexity.
- Avoid generating results using the same linguistic patterns.
""".lstrip(),
            }] + [{
                "role": "user",
                "content": f"Given the partial, \"{seed}\", generate {num_to_return} completed versions." + (
                    "" if not require_all_of else f" Use all of the following required words or phrases: {require_all_of}"
                ) + (
                    "" if not require_one_of else f" Use one of the following required words or phrases: {require_one_of}"
                ) + (
                    "" if not optional_components else f" Optionally, use one or more of the following words or phrases: {optional_components}"
                )
            }]),
            model=DEFAULT_CHAT_MODEL, n=1, temperature=1,
            tool_choice={
                "type": "function",
                "function": {"name": "return_sentence_completion_candidates"},
            },
            tools=[{
                "type": "function",
                "function": {
                    "name": "return_sentence_completion_candidates",
                    "description": " ".join((
                        "Returns generated list of completed sentence candidates to user.",
                    )),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "candidates": {
                                "type": "array",
                                "items": {
                                    "type": "string"
                                },
                                "description": "List of sentences.",
                            }
                        },
                        "required": ["candidates"]
                    },
                }
            }]
        )
    return json.loads(completion.choices[0].message.tool_calls[0].function.arguments)['candidates']


def singular_or_plural(word: str, count: int) -> str:
    if count > 1:
        return INFLECT_ENGINE.plural(word)
    else:
        singular_word = INFLECT_ENGINE.singular_noun(word)
        if singular_word:
            return singular_word
        else:
            return word  # If already singular or cannot determine, keep the original word


if __name__ == '__main__':
    print(json.dumps(sentence_completion_candidates("Generate an image of a small dog, sleeping", 10), indent=2))
