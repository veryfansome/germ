import difflib
import json
from nltk.tokenize import sent_tokenize
from openai import OpenAI

from bot.graph.noun import default_semantic_categories
from observability.logging import logging

differ = difflib.Differ()
logger = logging.getLogger(__name__)


class OpenAITextClassifier:

    TOOL_NAME = "store_text_classifications"

    def __init__(self, tool_properties_spec,
                 frequency_penalty: float = 0.0,
                 model: str = "gpt-4o-mini",
                 presence_penalty: float = 0.0,
                 system_message: str = "",
                 temperature: float = 0,
                 tool_name: str = "",
                 tool_description: str = "",
                 tool_parameter_description: str = ""):
        tool_name = OpenAITextClassifier.TOOL_NAME if not tool_name else tool_name
        self.frequency_penalty = frequency_penalty
        self.model = model
        self.presence_penalty = presence_penalty
        self.temperature = temperature

        self.system_message = (
            "Analyze the text, paying attention to word use and grammatical cues, "
            f"then generate classifications for the {tool_name} tool."
        ) if not system_message else system_message

        self.tool_description = "Store generated text classifications." if not tool_description else tool_description
        self.tool_name = tool_name
        self.tool_parameter_description = "Classifications to be stored." if not tool_parameter_description else tool_parameter_description
        self.tool_properties_spec = tool_properties_spec

    def classify(self, text: str, review: bool = False, review_json: str = None):
        tool_spec = {
            "type": "function",
            "function": {
                "name": self.tool_name,
                "description": self.tool_description,
                "parameters": {
                    "type": "object",
                    "description": self.tool_parameter_description,
                    "properties": self.tool_properties_spec,
                    "required": list(self.tool_properties_spec.keys()),
                    "additionalProperties": False,
                },
            }
        }
        messages = [{"role": "system", "content": self.system_message}]
        if review:
            messages.append({"role": "user", "content": (
                f"Review this classification and make correction as needed: {review_json}\n"
                f"Text: {text}"
            )})
        else:
            messages.append({"role": "user", "content": text})
        with OpenAI() as client:
            completion = client.chat.completions.create(
                messages=[{"role": "system", "content": self.system_message}, {"role": "user", "content": text}],
                model=self.model, n=1, timeout=30,
                frequency_penalty=self.frequency_penalty,
                presence_penalty=self.presence_penalty,
                temperature=self.temperature,
                tool_choice={"type": "function", "function": {"name": self.tool_name}},
                tools=[tool_spec])
            new_json = completion.choices[0].message.tool_calls[0].function.arguments
            if review and not review_json:
                return json.loads(self.classify(text, review=True, review_json=new_json))
            return json.loads(new_json)

    def get_tool_properties_spec(self):
        return self.tool_properties_spec


def get_noun_modifier_classifier(semantic_categories: list[str] = default_semantic_categories) -> OpenAITextClassifier:
    return OpenAITextClassifier({
        "modifiers": {
            "type": "array",
            "description": ("List of all modifier words in the the noun phrase, "
                            f"given that semantic categories include: {semantic_categories}."),
            "items": {
                "type": "object",
                "description": "A modifier word from the phrase.",
                "properties": {
                    "modifier": {
                        "type": "string",
                        "description": "Modifier word."
                    },
                    "noun": {
                        "type": "string",
                        "description": "What is the noun this modifier modify?"
                    },
                }
            }
        }},
        system_message=("Analyze the provided noun phrase as it appears in the text and look for the underlying "
                        "words being modified. Focus how modifier words in the phrase affects the text's meaning and "
                        "intention, then populate the modifiers parameter array for the store_modifiers tool."),
        tool_name="store_modifiers",
        tool_description="Store generated modifier classifications.",
        tool_parameter_description="Modifier classifications to be stored.",
    )


def get_sentence_nouns_classifier(semantic_categories: list[str] = default_semantic_categories) -> OpenAITextClassifier:
    return OpenAITextClassifier({
        "nouns": {
            "type": "array",
            "description": "List of all nouns identified in the text.",
            "items": {
                "type": "object",
                "description": "A noun from the text.",
                "properties": {
                    "noun": {
                        "type": "string",
                        "description": "Noun word."
                    },
                    "plurality": {
                        "type": "string",
                        "enum": ["singular", "plural"],
                    },
                    "semanticCategory": {
                        "type": "string",
                        "description": "What kind of thing is this noun?",
                        "enum": semantic_categories
                    },
                }
            }
        }},
        system_message=("Analyze the nouns from the text, focusing on how each affects meaning and intention, "
                        "then populate the nouns parameter array for the store_nouns tool."),
        tool_name="store_nouns",
        tool_description="Store generated noun classifications.",
        tool_parameter_description="Noun classifications to be stored.",
    )


def get_sentence_classifier(additional_parameters=None):
    return OpenAITextClassifier({
        "functional_type": {
            "type": "string",
            "description": ("Type of sentence based on function; is it stating facts or opinions, sometimes with "
                            "exclamations (declarative), does it ask a question or seek information, often ending in a "
                            "question mark (interrogative), or does it give a command, request, or instruction "
                            "(imperative)?"),
            "enum": ["declarative", "interrogative", "imperative"],
        },
        "organizational_type": {
            "type": "string",
            "description": "Type of sentence based on organization.",
            "enum": ["simple", "conditional", "compound and/or complex"],
        },
        **(additional_parameters if additional_parameters else {}),
    })


def get_single_noun_classifier(semantic_categories: list[str] = default_semantic_categories) -> OpenAITextClassifier:
    return OpenAITextClassifier({
        "plurality": {
            "type": "string",
            "enum": ["singular", "plural"],
        },
        "semanticCategory": {
            "type": "string",
            "description": "What kind of thing is this noun?",
            "enum": semantic_categories
        }},
        system_message=("Analyze the provided noun's use in the text, focusing on how it affects meaning and "
                        "intention, then populate the plurality and semanticCategory parameters for the "
                        "store_noun_classification tool."),
        tool_name="store_noun_classification",
        tool_description="Store generated noun classifications.",
        tool_parameter_description="Noun classifications to be stored.",
    )


def split_to_sentences(text: str) -> list[str]:
    return sent_tokenize(text)
