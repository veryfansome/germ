import difflib
import json
from nltk.tokenize import sent_tokenize
from openai import OpenAI

from bot.graph.entities import default_entity_types
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

# TODO: description classifier


emotion_to_entity_classifier = OpenAITextClassifier({
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


def get_entity_modifier_classifier(entity_types: list[str] = default_entity_types) -> OpenAITextClassifier:
    return OpenAITextClassifier({
        "modifiers": {
            "type": "array",
            "description": ("List of all modifier words in the the entity string, "
                            f"given that entity type categories include: {entity_types}."),
            "items": {
                "type": "object",
                "description": "A modifier word from the entity string.",
                "properties": {
                    "modifier": {
                        "type": "string",
                        "description": "Modifier word."
                    },
                    "modifies": {
                        "type": "string",
                        "description": "What word does this modifier modify?"
                    },
                }
            }
        }},
        system_message=("Analyze the provided entity string as it appears in the text and look for the underlying "
                        "words being modified. Focus how modifier words in the entity string affects the text's "
                        "meaning and intention, then populate the modifiers parameter array for the "
                        "store_modifiers tool."),
        tool_name="store_modifiers",
        tool_description="Store generated modifier classifications.",
        tool_parameter_description="Modifier classifications to be stored.",
    )


def get_entity_type_classifier(entity_types: list[str] = default_entity_types) -> OpenAITextClassifier:
    return OpenAITextClassifier({
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
                    "plurality": {
                        "type": "string",
                        "enum": ["singular", "plural"],
                    },
                    "entity_type": {
                        "type": "string",
                        "description": "What type or class of entity is this?",
                        "enum": entity_types
                    },
                }
            }
        }},
        system_message=("Analyze the entities from the text, focusing on how each affects meaning and intention, "
                        "then populate the entities parameter array for the store_entities tool."),
        tool_name="store_entities",
        tool_description="Store generated entity classifications.",
        tool_parameter_description="Entity classifications to be stored.",
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


def get_single_entity_type_classifier(entity_types: list[str] = default_entity_types) -> OpenAITextClassifier:
    return OpenAITextClassifier({
        "plurality": {
            "type": "string",
            "enum": ["singular", "plural"],
        },
        "entity_type": {
            "type": "string",
            "description": "What type or class of entity is this?",
            "enum": entity_types
        }},
        system_message=("Analyze the provided entity's use in the text, focusing on how it affects meaning and "
                        "intention, then populate the plurality and entity_type parameters for the store_entity_type "
                        "tool."),
        tool_name="store_entity_type",
        tool_description="Store generated entity type classifications.",
        tool_parameter_description="Entity type classifications to be stored.",
    )


entity_role_classifier = OpenAITextClassifier({
    "semantic_role": {
        "type": "string",
        "description": "Semantic role of the entity; does it perform the action (agent), is it affected "
                       "by the action or undergoes a state of change (patient), does it perceive or "
                       "experience the event or state (experiencer), is it moved by the action or whose "
                       "location is described (theme), is it the means by which an action is performed"
                       "(instrument), is it for this entity's benefit that the action is performed"
                       "(beneficiary), is it towards this entity which the action is directed or the "
                       "endpoint of a movement (goal), is it the entity from which something moves or "
                       "originates (source), is it the place where the action occurs (location), is it "
                       "the temporal context in which the action occurs (time), is it the way in which an "
                       "action is performed (manner), is it the reason or cause for action or state "
                       "(cause), does it receive something as a result of the action (recipient), is it "
                       "the entity that owns or possesses another entity (possessor), or is it the entity "
                       "that accompanies or is associated with another entity in the action (accompaniment)?",
        "enum": [
            "accompaniment" "agent", "beneficiary", "cause", "experiencer", "goal", "instrument",
            "location", "manner", "patient", "possessor", "recipient", "source", "time", "theme",
        ],
    }},
    system_message=("Analyze the entity's semantic role in the sentence, then generate the semantic_role parameter for "
                    "the store_semantic_role tool."),
    tool_name="store_semantic_role",
    tool_description="Store entity semantic role.",
    tool_parameter_description="Entity semantic role to be stored.",
    frequency_penalty=1,
)

entity_sentiment_classifier = OpenAITextClassifier({
    "sentiment_towards_entity": {
        "type": "string",
        "description": "Text's sentiment towards entity; clearly favorable (positive), clearly negative "
                       "(negative), objective, lacking emotional tone (neutral), contains positive and "
                       "negative sentiments (mixed).",
        "enum": ["mixed", "negative", "neutral", "positive"]
    }},
    system_message=("Analyze text's sentiment towards the entity, then generate the sentiment_towards_entity parameter "
                    "for the store_entity_sentiment tool."),
    tool_name="store_entity_sentiment",
    tool_description="Store entity sentiment.",
    tool_parameter_description="Entity sentiment to be stored.",
    frequency_penalty=1,
)

equivalence_classifier = OpenAITextClassifier({
    "equivalences": {
        "type": "array",
        "description": ("List of all equivalences, or X *is* Y type relationships, in the text, "
                        "inclusive of numerals where applicable."),
        "items": {
            "type": "object",
            "description": "An equivalence relationship.",
            "properties": {
                "X": {
                    "type": "string",
                    "description": "What is the term or thing being defined?",
                },
                "Y": {
                    "type": "string",
                    "description": "The complete definition, exclusive of conditions?",
                },
                "relationship_type": {
                    "type": "string",
                    "description": ("What type of equivalences is this; are X and Y the the same specific object "
                                    "(identical), do they have synonymous usages (synonym), is X a kind of Y "
                                    "(subset), is X a concept that is explained through Y (definition), or do X and "
                                    "Y have the same interchangeable value, measurement, or function (equivalent)?"),
                    "enum": ["identical", "synonym", "subset", "definition", "equivalent"]
                },
                "conditions": {
                    "type": "array",
                    "description": "List of contextual conditions for the equivalence X and Y.",
                    "items": {
                        "type": "string",
                        "description": "A condition in the text.",
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


def diff_strings(str1, str2):
    result = list(differ.compare(str1.splitlines(), str2.splitlines()))
    differences = [line for line in result if line.startswith('+ ') or line.startswith('- ')]
    return differences


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
