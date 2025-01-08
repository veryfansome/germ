from flair.data import Sentence
from flair.models import SequenceTagger
from nltk.tokenize import sent_tokenize
from openai import OpenAI
import difflib
import json
import re

from bot.graph.semantic_categories import default_semantic_categories
from observability.logging import logging

differ = difflib.Differ()
logger = logging.getLogger(__name__)
flair_pos_tagger = SequenceTagger.load("pos")

ADJECTIVE_POS_TAGS = set()
ADJECTIVE_POS_TAGS.add("JJ")  # Standard, "big"
ADJECTIVE_POS_TAGS.add("JJR")  # Comparative, "bigger"
ADJECTIVE_POS_TAGS.add("JJS")  # Superlative, "biggest"

ADVERB_POS_TAGS = set()
ADVERB_POS_TAGS.add("RB")  # Standard, "quickly", "well"
ADVERB_POS_TAGS.add("RBR")  # Comparative, "more quickly", "better"
ADVERB_POS_TAGS.add("RBS")  # Superlative, "most quickly", "best"

NOUN_POS_TAGS = set()
NOUN_POS_TAGS.add("NN")  # Singular or mass
NOUN_POS_TAGS.add("NNS")  # Plural
NOUN_POS_TAGS.add("NNP")  # Singular proper noun
NOUN_POS_TAGS.add("NNPS")  # Plural proper noun

PRONOUN_POS_TAGS = set()
PRONOUN_POS_TAGS.add("PRP")  # Personal
PRONOUN_POS_TAGS.add("PRP$")  # Possessive
PRONOUN_POS_TAGS.add("WP")  # Wh-pronoun
PRONOUN_POS_TAGS.add("WP$")  # Possessive wh-pronoun

VERB_POS_TAGS = set()
VERB_POS_TAGS.add("VB")  # Base form, "to run"
VERB_POS_TAGS.add("VBD")  # Past tense, "ran"
VERB_POS_TAGS.add("VBG")  # Gerund or present participle, "running"
VERB_POS_TAGS.add("VBN")  # past participle, "run"
VERB_POS_TAGS.add("VBP")  # non-3rd person singular present, "run"
VERB_POS_TAGS.add("VBZ")  # 3rd person singular present, "runs"


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


def get_flair_pos_tags(sentence_text: str):
    if "<code>" in sentence_text:
        preprocessed_text, code_snippets = preprocess_text_for_pos_tagging(sentence_text)
        sentence = Sentence(preprocessed_text)
        flair_pos_tagger.predict(sentence)
        return postprocess_sentence_pos_tags(sentence, code_snippets)
    else:
        sentence = Sentence(sentence_text)
        flair_pos_tagger.predict(sentence)
        return [(word.text, word.tag) for word in sentence]


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


def postprocess_sentence_pos_tags(sentence: Sentence, code_snippets: list[str]):
    pos_tags = []
    snippet_map = {}
    # Map snippets to placeholders
    for i, snippet in enumerate(code_snippets):
        # TODO: flair isn't recognizing the placeholder format - try CODE1 or some single token we can easily
        placeholder = f"<CODE_SNIPPET_{i}>"
        snippet_map[placeholder] = snippet
    # Replace placeholders back with original code snippets
    for word in sentence:
        if word.text.startswith("<CODE_SNIPPET_"):
            pos_tags.append((snippet_map[word.text], "CODE"))
        else:
            pos_tags.append((word.text, word.tag))
    return pos_tags


def preprocess_text_for_pos_tagging(text: str):
    # Replace <code>...</code> with a placeholder
    code_snippets = re.findall(r'<code>(.*?)</code>', text)
    for i, snippet in enumerate(code_snippets):
        placeholder = f"<CODE_SNIPPET_{i}>"
        text = text.replace(f"<code>{snippet}</code>", placeholder)
    return text, code_snippets


def split_to_sentences(text: str) -> list[str]:
    return sent_tokenize(text)
