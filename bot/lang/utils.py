import difflib
import json
from nltk.tokenize import sent_tokenize
from openai import OpenAI

from observability.logging import logging

differ = difflib.Differ()
logger = logging.getLogger(__name__)


class OpenAISentenceClassifier:

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
        tool_name = OpenAISentenceClassifier.TOOL_NAME if not tool_name else tool_name
        self.frequency_penalty = frequency_penalty
        self.model = model
        self.presence_penalty = presence_penalty
        self.system_message = f"Classify this text based on the parameters for the {tool_name} tool." if not system_message else system_message
        self.temperature = temperature
        self.tool_description = "Store generated text classifications." if not tool_description else tool_description
        self.tool_name = tool_name
        self.tool_parameter_description = "Classifications to be stored." if not tool_parameter_description else tool_parameter_description
        self.tool_properties_spec = tool_properties_spec

    def classify(self, text: str, review: bool = False, _review_json: str = None):
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
                f"Review this classification and make correction as needed: {_review_json}\n"
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
            if review and not _review_json:
                self.classify(text, review=True, _review_json=new_json)
            return json.loads(new_json)


# TODO: description classifier


emotion_to_entity_classifier = OpenAISentenceClassifier({
    "emotions": {
        "type": "array",
        "description": "List of all perceivable emotions, capturing nuanced contextual meaning.",
        "items": {
            "type": "object",
            "description": "An emotion from the text.",
            "properties": {
                "emotion": {
                    "type": "string",
                    "description": "Detected emotion name."
                },
                "felt_by": {
                    "type": "string",
                    "description": "Who feels this emotion?.",
                },
                "felt_by_entity_type": {
                    "type": "string",
                    "description": "What kind of entity is the feeler of this emotion.",
                },
                "felt_towards": {
                    "type": "string",
                    "description": "What is this emotion felt towards?",
                },
                "felt_towards_entity_type": {
                    "type": "string",
                    "description": "What kind of entity is this emotion felt towards?",
                },
            }
        }
    }},
    system_message=("Analyze the emotional nuance from the text "
                    f"and populate the emotions array for the store_emotions tool."),
    tool_name="store_emotions",
    tool_description="Store generated emotion classifications.",
    tool_parameter_description="Emotion classifications to be stored.",
    frequency_penalty=1,
)


equivalence_classifier = OpenAISentenceClassifier({
    "equivalences": {
        "type": "array",
        "description": ("List of all equivalences, or X *is* Y type relationships, mentioned in the text, "
                        "inclusive of numerals where applicable."),
        "items": {
            "type": "object",
            "description": "An equivalence relationship.",
            "properties": {
                "X": {
                    "type": "string",
                    "description": "What is the X?",
                },
                "Y": {
                    "type": "string",
                    "description": "What is the Y?",
                },
                "relationship_type": {
                    "type": "string",
                    "description": ("What type of equivalences is this; are X and Y the the same specific object "
                                    "(identical), do they have synonymous usages (synonym), is X a kind of Y "
                                    "(subset), is X a concept that should be understood relative to concept Y "
                                    "(definition), or do X and Y have the same interchangeable value, measurement, or "
                                    "function (equivalent)?"),
                    "enum": ["identical", "synonym", "subset", "definition", "equivalent"]
                },
                "conditions": {
                    "type": "array",
                    "description": "List of conditions for equivalence.",
                    "items": {
                        "type": "string",
                        "description": "A condition spelled out in the text.",
                    }
                },
            }
        }
    }},
    system_message=("Analyze the equivalences in the text "
                    f"and populate the equivalences array for the store_equivalences tool."),
    tool_name="store_equivalences",
    tool_description="Store generated equivalence classifications.",
    tool_parameter_description="Equivalences classifications to be stored.",
    frequency_penalty=1,
)


def classify_emotions_using_openai(text: str,
                                   model: str = "gpt-4o-mini"):
    tool_name = "store_emotional_features"
    system_message = ("Analyze the emotional nuance from the text "
                      f"and populate the emotions array for the {tool_name} tool.")
    tool_properties_spec = {
        "emotions": {
            "type": "array",
            "description": "List of all perceivable emotions, capturing nuanced contextual meaning.",
            "items": {
                "type": "object",
                "description": "An emotion from the text.",
                "properties": {
                    "emotion": {
                        "type": "string",
                        "description": "Detected emotion name."
                    },
                    "felt_by": {
                        "type": "string",
                        "description": "Who feels this emotion?.",
                    },
                    "felt_by_entity_type": {
                        "type": "string",
                        "description": "What kind of entity is the source of this emotion.",
                    },
                    "felt_towards": {
                        "type": "string",
                        "description": "What is this emotion directed at?",
                    },
                    "felt_towards_entity_type": {
                        "type": "string",
                        "description": "What kind of entity is the target of this emotion?",
                    },
                    "intensity": {
                        "type": "string",
                        "description": "Intensity of detected emotion.",
                        "enum": ["low", "medium", "high"]
                    },
                    "nuance": {
                        "type": "string",
                        "description": "Emotion complexity.",
                        "enum": ["simple", "complex"]
                    },
                    "synonymous_emotions": {
                        "type": "array",
                        "description": "List of synonymous emotions based on context.",
                        "items": {
                            "type": "string",
                            "description": "Name of a synonymous emotion."
                        }
                    },
                    "opposite_emotions": {
                        "type": "array",
                        "description": "List of opposite emotions based on context.",
                        "items": {
                            "type": "string",
                            "description": "Name of an opposite emotion."
                        }
                    },
                }
            }
        }
    }
    with OpenAI() as client:
        completion = client.chat.completions.create(
            messages=[{"role": "system", "content": system_message}, {"role": "user", "content": text}],
            model=model, n=1, temperature=0, timeout=30,

            # -2 to 2, lower values to stay more focused on the text vs using different words
            # frequency_penalty=0,

            # -2 to 2, lower values discourage new topics
            # presence_penalty=0,

            tool_choice={"type": "function", "function": {"name": tool_name}},
            tools=[{
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": "Store text emotion features.",
                    "parameters": {
                        "type": "object",
                        "description": "Emotion features extracted from text.",
                        "properties": tool_properties_spec,
                        "required": list(tool_properties_spec.keys()),
                        "additionalProperties": False,
                    },
                }
            }]
        )
        return json.loads(completion.choices[0].message.tool_calls[0].function.arguments)


def classify_sentence_using_openai(text: str,
                                   model: str = "gpt-4o-mini"):
    tool_name = "store_sentence_classification"
    system_message = f"Classify this sentence and generate parameters for the {tool_name} tool."
    tool_properties_spec = {
        "functional_type": {
            "type": "string",
            "description": "Type of sentence based on function",
            "enum": ["conditional", "exclamatory", "declarative", "interrogative", "imperative"],
        },
        "organizational_type": {
            "type": "string",
            "description": "Type of sentence based on organization.",
            "enum": ["simple", "compound and/or complex"],
        },
        "change_in_state": {
            "type": "string",
            "description": "Does this sentence describe state changes?",
            "enum": ["yes", "no"],
        },
        "noticeable_emotions": {
            "type": "string",
            "description": "Does the sentence include non-neutral emotions?",
            "enum": ["yes", "no"],
        },
        "reports_speech": {
            "type": "string",
            "description": "Does the sentence report direct or paraphrased speech?",
            "enum": ["yes", "no"],
        },
        "spatiality": {
            "type": "string",
            "description": "Does the sentence describe spatial relationships?",
            "enum": ["yes", "no"],
        },
        "temporality": {
            "type": "string",
            "description": "Does the sentence describe temporal relationships?",
            "enum": ["yes", "no"],
        },
        "uses_jargon": {
            "type": "string",
            "description": "Does the sentence use jargon or slang?",
            "enum": ["yes", "no"],
        },
    }
    with OpenAI() as client:
        completion = client.chat.completions.create(
            messages=[{"role": "system", "content": system_message}, {"role": "user", "content": text}],
            model=model, n=1, temperature=0, timeout=30,
            # -2 to 2, lower values to stay more focused on the text vs using different words
            frequency_penalty=0,
            # -2 to 2, lower values discourage new topics
            presence_penalty=0,
            tool_choice={"type": "function", "function": {"name": tool_name}},
            tools=[{
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": "Store sentence classification.",
                    "parameters": {
                        "type": "object",
                        "description": "Sentence classification to store.",
                        "properties": tool_properties_spec,
                        "required": list(tool_properties_spec.keys()),
                        "additionalProperties": False,
                    },
                }
            }]
        )
        return json.loads(completion.choices[0].message.tool_calls[0].function.arguments)


def diff_strings(str1, str2):
    result = list(differ.compare(str1.splitlines(), str2.splitlines()))
    differences = [line for line in result if line.startswith('+ ') or line.startswith('- ')]
    return differences


def extract_openai_entity_features(text: str,
                                   model: str = "gpt-4o-mini"):
    tool_name = "store_entity_features"
    system_message = ("Analyze all entities from the text "
                      f"and populate the entities array for the {tool_name} tool.")
    tool_properties_spec = {
        "entities": {
            "type": "array",
            "description": "List of all entities identified in the text.",
            "items": {
                "type": "object",
                "description": "An entity from the text.",
                "properties": {
                    "entity": {
                        "type": "string",
                        "description": "Entity name."
                    },
                    "entity_type": {
                        "type": "string",
                        "description": "What type or class of entity is this?",
                        "enum": [
                            "event in time",
                            "action or possible action",
                            "concept", "creature", "currency",
                            "geographic feature",
                            "location",
                            "organization",
                            "person or personified being", "point in time", "possession object",
                            "quantity",
                            "structure",
                        ]
                    },
                    "sentiment": {
                        "type": "string",
                        "description": "Sentiment towards entity in text.",
                        "enum": ["mixed", "negative", "neutral", "positive"]
                    },
                    "semantic_role": {
                        "type": "string",
                        "description": "Semantic role of entity in text.",
                    },
                }
            }
        }
    }
    with OpenAI() as client:
        completion = client.chat.completions.create(
            messages=[{"role": "system", "content": system_message}, {"role": "user", "content": text}],
            model=model, n=1, temperature=0, timeout=30,

            # -2 to 2, lower values to stay more focused on the text vs using different words
            # frequency_penalty=0,

            # -2 to 2, lower values discourage new topics
            # presence_penalty=0,

            tool_choice={"type": "function", "function": {"name": tool_name}},
            tools=[{
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": "Store text entity features.",
                    "parameters": {
                        "type": "object",
                        "description": "Entity features extracted from text.",
                        "properties": tool_properties_spec,
                        "required": list(tool_properties_spec.keys()),
                        "additionalProperties": False,
                    },
                }
            }]
        )
        return json.loads(completion.choices[0].message.tool_calls[0].function.arguments)


def extract_openai_speech_features(text: str, model: str = "gpt-4o-mini"):
    tool_name = "store_speech_features"
    system_message = f"Analyze the speech in this text and populate the speech parameter for the {tool_name} tool."
    tool_properties_spec = {
        "speech": {
            "type": "array",
            "description": "List of all instances of reported speech found in text.",
            "items": {
                "type": "object",
                "description": "An instance of reported speech from the text.",
                "properties": {
                    "speech": {
                        "type": "string",
                        "description": "Direct speech text or paraphrased text.",
                    },
                    "speech_type": {
                        "type": "string",
                        "description": "Type of speech.",
                        "enum": ["first-person direct", "second-person direct", "third-person direct",
                                 "first-person paraphrased", "second-person paraphrased", "third-person paraphrased"]
                    },
                    "narrator": {
                        "type": "string",
                        "description": "Who is the narrator?",
                    },
                    "listener": {
                        "type": "string",
                        "description": "Who is the listener?",
                    },
                    "speech_owner": {
                        "type": "string",
                        "description": "Who is responsible for the reported speech?",
                    },
                }
            }
        },
    }
    with OpenAI() as client:
        completion = client.chat.completions.create(
            messages=[{"role": "system", "content": system_message}, {"role": "user", "content": text}],
            model=model, n=1, temperature=0, timeout=30,
            # -2 to 2, lower values to stay more focused on the text vs using different words
            frequency_penalty=0,
            # -2 to 2, lower values discourage new topics
            presence_penalty=0,
            tool_choice={"type": "function", "function": {"name": tool_name}},
            tools=[{
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": "Store speech features.",
                    "parameters": {
                        "type": "object",
                        "description": "Speech features to store.",
                        "properties": tool_properties_spec,
                        "required": list(tool_properties_spec.keys()),
                        "additionalProperties": False,
                    },
                }
            }]
        )
        return json.loads(completion.choices[0].message.tool_calls[0].function.arguments)


def extract_openai_state_change_features(text: str, model: str = "gpt-4o-mini"):
    tool_name = "store_state_change_features"
    system_message = ("Analyze the changes in state mentioned or implied in this text "
                      f"and populate the state_changes parameter for the {tool_name} tool.")
    tool_properties_spec = {
        "state_changes": {
            "type": "array",
            "description": ("List of all changes in state related to entities, objects, subjects, "
                            "or other elements of the text."),
            "items": {
                "type": "object",
                "description": "An instance of something changing from one state to another in the text.",
                "properties": {
                    "what": {
                        "type": "string",
                        "description": "What is the thing that changes?",
                    },
                    "state_before_change": {
                        "type": "string",
                        "description": "State before change.",
                    },
                    "state_after_change": {
                        "type": "string",
                        "description": "State after change.",
                    },
                    "change_process": {
                        "type": "string",
                        "description": "Name or word that best describes the change process.",
                    },
                    "spatiality": {
                        "type": "string",
                        "description": "Where does the change occur?",
                        "enum": ["current location", "other unspecified location", "other specified location"],
                    },
                    "temporality": {
                        "type": "string",
                        "description": "When does the change occur?",
                        "enum": ["past", "present", "future"],
                    },
                    "change_agent": {
                        "type": "string",
                        "description": "What is the agent that realizes the change?.",
                    },
                }
            }
        },
    }
    with OpenAI() as client:
        completion = client.chat.completions.create(
            messages=[{"role": "system", "content": system_message}, {"role": "user", "content": text}],
            model=model, n=1, temperature=0, timeout=30,
            # -2 to 2, lower values to stay more focused on the text vs using different words
            frequency_penalty=0,
            # -2 to 2, lower values discourage new topics
            presence_penalty=0,
            tool_choice={"type": "function", "function": {"name": tool_name}},
            tools=[{
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": "Store state change features.",
                    "parameters": {
                        "type": "object",
                        "description": "State change features to store.",
                        "properties": tool_properties_spec,
                        "required": list(tool_properties_spec.keys()),
                        "additionalProperties": False,
                    },
                }
            }]
        )
        return json.loads(completion.choices[0].message.tool_calls[0].function.arguments)


def extract_openai_text_features(text: str, model: str = "gpt-4o-mini"):
    """
    Way better for emotionality (variety, intensity, transitions), sentiment, related knowledge areas, related topics,
    and other contextual tasks.

    Setting `second_opinion` to True can improve accuracy but is obviously more expensive.

    :param text:
    :param model:
    :return:
    """
    system_message = "Extract text features."

    tool_name = "store_text_features"
    tool_properties_spec = {
        "knowledge": {
            "type": "array",
            "description": "List of applicable knowledge categories.",
            "items": {
                "type": "string",
                "description": "A category."
            }
        },
        "sentiment": {
            "type": "string",
            "description": "Sentiment of text.",
            "enum": ["mixed", "negative", "neutral", "positive"]
        },
        "subjects": {
            "type": "array",
            "description": "List of sentence subjects.",
            "items": {
                "type": "string",
                "description": "A subject."
            }
        },
        "topics": {
            "type": "array",
            "description": "List of topics.",
            "items": {
                "type": "string",
                "description": "A topic."
            }
        },
    }
    with OpenAI() as client:
        completion = client.chat.completions.create(
            messages=[{"role": "system", "content": system_message}, {"role": "user", "content": text}],
            model=model, n=1, temperature=0, timeout=60,

            # -2 to 2, lower values to stay more focused on the text vs using different words
            # frequency_penalty=0,

            # -2 to 2, lower values discourage new topics
            # presence_penalty=0,

            tool_choice={"type": "function", "function": {"name": tool_name}},
            tools=[{
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": "Store extracted text features.",
                    "parameters": {
                        "type": "object",
                        "description": "Features extracted from text.",
                        "properties": tool_properties_spec,
                        "required": list(tool_properties_spec.keys()),
                        "additionalProperties": False,
                    },
                }
            }]
        )
        return json.loads(completion.choices[0].message.tool_calls[0].function.arguments)


def split_to_sentences(text: str) -> list[str]:
    return sent_tokenize(text)
