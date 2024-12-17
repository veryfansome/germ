import re
from flair.data import Sentence
from flair.models import SequenceTagger
from openai import OpenAI

from observability.logging import setup_logging

ner_tagger = SequenceTagger.load("ner")


def findall_regex_words(sentence: str) -> list[str]:
    return re.findall(r'\w+', sentence)


def flair_ner(sentence: Sentence):
    ner_tagger.predict(sentence)
    return sentence.get_spans('ner')


def openai_text_feature_extraction(text: str):
    with OpenAI() as client:
        completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "Extract text features.",
                },
                {
                    "role": "user",
                    "content": text,
                },
            ],
            model="gpt-4o-mini",
            n=1, temperature=0,
            timeout=60,
            tool_choice={
                "type": "function",
                "function": {"name": "store_text_features"}
            },
            tools=[{
                "type": "function",
                "function": {
                    "name": "store_text_features",
                    "description": "Store extracted text features.",
                    "parameters": {
                        "type": "object",
                        "description": "Features extracted from text.",
                        "properties": {
                            "emotions": {
                                "type": "array",
                                "description": "List of emotions.",
                                "items": {
                                    "type": "string",
                                    "description": "An emotion.",
                                }
                            },
                            "emotional_intensity": {
                                "type": "string",
                                "enum": ["low", "high"]
                            },
                            "entities": {
                                "type": "array",
                                "description": "List of named entities.",
                                "items": {
                                    "type": "object",
                                    "description": "A named entity.",
                                    "properties": {
                                        "name": {
                                            "type": "string",
                                            "description": "Named entity name."
                                        },
                                        "tag": {
                                            "type": "string",
                                            "description": "Named entity tag like: PER, ORG, LOC, etc."
                                        },
                                    }
                                }
                            },
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
                        },
                        "required": [
                            "emotions", "emotional_intensity", "entities", "knowledge",
                            "sentiment", "subjects", "topics",
                        ],
                        "additionalProperties": False,
                    },
                }
            }]
        )
        return completion.choices[0].message.tool_calls[0].function.arguments


def text_to_punctuation_blob(text: str):
    return ''.join(re.findall(r'[^a-zA-Z0-9\s]+', text))


if __name__ == '__main__':
    setup_logging(global_level="ERROR")

    test_set = [
        "Hello, world!",
        "The Earth is round?",
        "This sucks!",
        "I hated broccoli as a kid but now I love it.",
        "Clifford is the big red dog.",
        "I just love beyonce.",
        "Cut my hair just like Beyonce",
        "It's hard to believe my husband used to be such a rude person.",
        "Nevermind is my favorite Nirvana album",
        "People might just be nice for their gain.",
        "People may have complex motivations that require discernment.",
        "Maybe people bring joy, but not always in ways I expect.",
        "People might be self-interested; goodness could just be a disguise.",
        "People might surprise us with kindness when we least expect it.",
        "People might just be self-serving in the end.",
    ]
    for test in test_set:
        flair_sentence = Sentence(test)
        print(f"""
Sentence: {test}
---
findall_regex_words: {findall_regex_words(test)}
flair_ner: {flair_ner(flair_sentence)}
openai_text_feature_extraction: {openai_text_feature_extraction(test)}
text_to_punctuation_blob: {text_to_punctuation_blob(test)}
""")
