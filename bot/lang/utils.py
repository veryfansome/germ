import difflib
from nltk.tokenize import sent_tokenize
from openai import OpenAI

from observability.logging import logging

differ = difflib.Differ()
logger = logging.getLogger(__name__)


def diff_strings(str1, str2):
    result = list(differ.compare(str1.splitlines(), str2.splitlines()))
    differences = [line for line in result if line.startswith('+ ') or line.startswith('- ')]
    return differences


def extract_openai_emotion_features(text: str,
                                    model: str = "gpt-4o-mini") -> str:
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
                    "emotion_source": {
                        "type": "string",
                        "description": "Who feels this emotion?.",
                    },
                    "emotion_source_entity_type": {
                        "type": "string",
                        "description": "What kind of entity is the source of this emotion.",
                    },
                    "emotion_target": {
                        "type": "string",
                        "description": "What is this emotion directed at?",
                    },
                    "emotion_target_entity_type": {
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
        return completion.choices[0].message.tool_calls[0].function.arguments


def extract_openai_entity_features(text: str,
                                   model: str = "gpt-4o-mini") -> str:
    system_message = "Extract entities from text."
    tool_name = "store_entity_features"
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
        return completion.choices[0].message.tool_calls[0].function.arguments


def extract_openai_sentence_type_features(text: str,
                                          model: str = "gpt-4o-mini"):
    system_message = "Detect sentence type from textual context."
    tool_name = "store_sentence_type"
    tool_properties_spec = {
        "change_of_state": {
            "type": "string",
            "description": "Do the subjects or objects in this sentence undergo a change in state?",
            "enum": ["complex changes", "objects change", "static", "subjects change"],
        },
        "contains_interjection": {
            "type": "string",
            "description": "Does the sentence include a sudden direction change, often standing alone?",
            "enum": ["true", "false"],
        },
        "functional_type": {
            "type": "string",
            "description": "Type of sentence based on purpose.",
            "enum": ["conditional", "exclamatory", "declarative", "interrogative", "imperative"],
        },
        "narrative_structure": {
            "type": "string",
            "description": "Does the sentence try to tell a story or is it providing descriptions?",
            "enum": ["event-driven", "detail-driven"],
        },
        "organizational_type": {
            "type": "string",
            "description": "Type of sentence based on organization.",
            "enum": ["simple", "compound", "complex", "complex-compound"],
        },
        "reports_speech": {
            "type": "string",
            "description": "How does the sentence report speech?",
            "enum": ["no speech", "direct speech", "paraphrased speech"],
        },
        "spatiality": {
            "type": "string",
            "description": "Do the subjects or objects in this sentence move through space?",
            "enum": ["complex movements", "objects move", "static", "subjects move"],
        },
        "style": {
            "type": "string",
            "description": ("Does the sentence adhere to standard language rules (formal) "
                            "or does it include colloquialisms or slang (informal)"),
            "enum": ["formal", "informal"],
        },
        "temporality": {
            "type": "string",
            "description": "Does time progress in this sentence or is it static?",
            "enum": ["past to past", "past to present", "present to future",
                     "static future", "static past", "static present"],
        },
        "voice": {
            "type": "string",
            "description": "Does the subject perform the action (active) or receive the action (passive)?",
            "enum": ["active", "passive"],
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
                    "description": "Store sentence type.",
                    "parameters": {
                        "type": "object",
                        "description": "Sentence type to store.",
                        "properties": tool_properties_spec,
                        "required": list(tool_properties_spec.keys()),
                        "additionalProperties": False,
                    },
                }
            }]
        )
        return completion.choices[0].message.tool_calls[0].function.arguments


def extract_openai_text_features(text: str, model: str = "gpt-4o-mini") -> str:
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
        return completion.choices[0].message.tool_calls[0].function.arguments


def split_to_sentences(text: str) -> list[str]:
    return sent_tokenize(text)
