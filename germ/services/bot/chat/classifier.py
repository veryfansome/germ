import json
import logging
from traceback import format_exc

from germ.services.bot.chat import async_openai_client
from germ.settings import germ_settings

logger = logging.getLogger(__name__)


class UserIntentClassifier:

    label_spec = {
        "judgment": {
            "description": "user's goal is to convey a decision",
            "labels": {
                "accepts as true": "user accepts statement(s) as true",
                "rejects as false": "user rejects statement(s) as false",
                "consents": "user gives consent",
                "refuses consent": "user denies or withholds consent",
            }
        },
        "informational": {
            "description": "user's goal is to receive information",
            "labels": {
                "fact-seeking": "user wants a piece of factual information",
                "clarification": "user is asking for something to be explained",
                "education": "user is looking to learn about a topic in detail",
            }
        },
        "relational": {
            "description": "user's goal is to receive emotional support, socialize, or establish trust",
            "labels": {
                "emotional support": "user is looking for comfort or empathy",
                "social bonding": "user wants to chat and build a connection",
                "validation": "user is looking for acknowledgement of their feelings or experience",
            }
        },
        "instrumental": {
            "description": "user's goal is to get something done",
            "labels": {
                "problem-solving": "user wants help with a specific issue",
                "task-oriented request": "user is asking for something to be done step-by-step",
                "decision support": "user needs help making a choice or weighing options",
            }
        },
        "expressive": {
            "description": "user's goal is to express feelings, opinions, or identity",
            "labels": {
                "self-expression": "user wants to share their thoughts or feelings",
                "opinion-sharing": "user wants to convey their thoughts about a certain topic",
                "creative expression": "user wants to be imaginative or playful",
            }
        },
        "miscellaneous": {
            "description": "user's goal doesn't fit in the other categories",
            "labels": {
                "other": "what the user wants does not fit a defined intent label",
                "unclear": "what the user wants is not clear",
            }
        },
    }

    label_category_descriptions = ''.join(f"\n - {k}: {v['description']}" for k, v in label_spec.items())

    label_descriptions = ''.join(f"\n - {k}: {l}, {d}" for k, v in label_spec.items() for l, d in v["labels"].items())

    labels = [f"{k}: {l}" for k, v in label_spec.items() for l in v["labels"].keys()]

    prompt = (
        "You are an insightful observer of human interactions and judge of user intentions. "

        "Consider the user's objective and return all intent labels that apply to the user's most recent message. "
        
        "Multiple labels may be appropriate but use only the labels described below. "
        
        "Do not invent or generalize beyond them. "

        f"\n\nEach label falls into one of six broad categorical buckets: "
        f"{label_category_descriptions}"
        
        "\n\nEach label is formated as '<category>: <intent>': "
        f"{label_descriptions}\n"
    ).strip()

    @classmethod
    async def suggest_user_intent(
            cls, messages: list[dict[str, str]],
    ) -> dict[str, list[str]]:
        intents: list[str] = []
        while not intents:
            try:
                response = await async_openai_client.chat.completions.create(
                    messages=[{"role": "system", "content": cls.prompt}] + messages,
                    model=germ_settings.CLASSIFICATION_MODEL,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "intent",
                            "schema": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "labels": {
                                        "type": "array",
                                        "description": "A list of user intention classifications.",
                                        "items": {
                                            "type": "string",
                                            "enum": cls.labels,
                                        },
                                        "minItems": 1,
                                        "uniqueItems": True,
                                    }
                                },
                                "required": ["labels"],
                            }
                        }
                    },
                    n=1,
                    timeout=15,
                )
                response_content = json.loads(response.choices[0].message.content)
                assert "labels" in response_content, "Response does not contain 'labels'"
                intents.extend(response_content["labels"])
            except Exception:
                logger.error(f"Could not get user intent suggestions: {format_exc()}")
        return {"intents": intents}
