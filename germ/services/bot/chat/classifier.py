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

    label_category_descriptions = ''.join(f"\n- {k}: {v['description']}" for k, v in label_spec.items())

    label_descriptions = ''.join(f"\n- {k}: {l}, {d}" for k, v in label_spec.items() for l, d in v["labels"].items())

    labels = [f"{k}: {l}" for k, v in label_spec.items() for l in v["labels"].keys()]

    prompt = (
        "You are a classifier of user intentions. "

        "\n\nTask: "
        "\n- Consider the user's objective and return all intent labels that apply to the user's most recent message. "

        "\n\nCategory descriptions: "
        f"{label_category_descriptions}"
        
        "\n\nIntent descriptions: "
        f"{label_descriptions}\n"

        "\n\nGuidelines: "
        "\n- Each label is formated as '<category>: <intent>'."
        "\n- Multiple labels may be appropriate but use only the labels described above. Do not invent or generalize beyond them. "
        "\n- Only use 'miscellaneous: unclear' when no other label applies. Don't use this label in combination with other labels."

        "\n\nOutput: "
        "\n- Return only a JSON object conforming to the provided schema with a 'labels' attribute that is a list of the labels described above. "
    ).strip()

    @classmethod
    async def classify_user_intent(
            cls, messages: list[dict[str, str]],
    ) -> dict[str, list[tuple[str, str]]]:
        intents: list[tuple[str, str]] = []
        while not intents:
            try:
                response = await async_openai_client.chat.completions.create(
                    messages=[{"role": "system", "content": cls.prompt}] + messages,
                    model=germ_settings.OPENAI_CLASSIFICATION_MODEL,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "intent",
                            "schema": {
                                "type": "object",
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
                                "additionalProperties": False,
                                "required": ["labels"],
                            },
                        }
                    },
                    n=1,
                    seed=germ_settings.OPENAI_SEED,
                    timeout=15.0,
                )
                response_content = json.loads(response.choices[0].message.content)
                assert "labels" in response_content, "Response does not contain 'labels'"
                labels: list[tuple[str, str]] = []
                for label_blobs in response_content["labels"]:
                    label_parts = label_blobs.lower().split(": ")
                    assert len(label_parts) == 2, f"Unexpected label format '{label_blobs}'"
                    labels.append((label_parts[0], label_parts[1]))
                intents.extend(labels)
            except Exception:
                logger.error(f"Could not get user intent suggestions: {format_exc()}")
        return {"intents": intents}
