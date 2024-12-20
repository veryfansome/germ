import difflib
from flair.data import Sentence
from nltk.tokenize import sent_tokenize
from openai import OpenAI

from bot.lang.dependencies import pos_tagger, ner_tagger
from observability.logging import logging, setup_logging

logger = logging.getLogger(__name__)

differ = difflib.Differ()


def diff_strings(str1, str2):
    result = list(differ.compare(str1.splitlines(), str2.splitlines()))
    differences = [line for line in result if line.startswith('+ ') or line.startswith('- ')]
    return differences


def flair_text_feature_extraction(text: str):
    """
    Old fashion NLP on the cheap.

    :param text:
    :return:
    """
    pos_tags = []
    proper_nouns = []
    verbs = []
    flair_sentence = Sentence(text)
    pos_tagger.predict(flair_sentence)
    ner_tagger.predict(flair_sentence)
    for token in flair_sentence:
        pos_tags.append(token.tag)
        if token.tag in ("NNP", "NNPS"):
            proper_nouns.append(token.text)
        elif token.tag in ("VBD", "VBP", "VBZ"):
            verbs.append(token.text)
    flair_features = {
        "ner": flair_sentence.get_spans("ner"),
        "pos_blob": "_".join(pos_tags),
        "proper_nouns": proper_nouns,
        "verbs": verbs}
    logger.info(f"flair_features: {flair_features}")
    return flair_features


def openai_text_feature_extraction(text: str,
                                   json_to_check: str = None,
                                   model: str = "gpt-4o-mini",
                                   prefer_second_opinion: bool = False,
                                   second_opinion: bool = False,
                                   temperature: int = 0) -> str:
    """
    Way better for emotionality (variety, intensity, transitions), sentiment, related knowledge areas, related topics,
    and other contextual tasks.

    Setting `second_opinion` to True can improve accuracy but is obviously more expensive.

    :param json_to_check:
    :param model:
    :param prefer_second_opinion:
    :param second_opinion:
    :param temperature:
    :param text:
    :return:
    """
    system_message = "Extract text features."
    if json_to_check is not None:
        system_message = "Check the extracted text features and correct errors."
        text = f"# Text\n{text}\n\n# Features JSON\n{json_to_check}"

    tool_name = "store_text_features"
    tool_properties_spec = {
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
            "description": "Intensity of emotions.",
            "enum": ["low", "medium", "high"]
        },
        "emotional_nuance": {
            "type": "string",
            "description": "Complexity of emotions.",
            "enum": ["low", "medium", "high"]
        },
        "entities": {
            "type": "array",
            "description": "List of named entities.",
            "items": {
                "type": "object",
                "description": "A named entity.",
                "properties": {
                    "entity": {
                        "type": "string",
                        "description": "Entity name."
                    },
                    "type": {
                        "type": "string",
                        "description": "Type of named entity."
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
        "sentence_type": {
            "type": "string",
            "description": "Sentence type.",
            "enum": ["complex", "conditional", "exclamatory", "declarative", "interrogative", "imperative"],
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
            model=model, n=1, temperature=temperature, timeout=60,
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
        new_feature_json = completion.choices[0].message.tool_calls[0].function.arguments
        if second_opinion:
            return openai_text_feature_extraction(
                text, json_to_check=new_feature_json, prefer_second_opinion=prefer_second_opinion)

        if json_to_check is None:
            logger.info(f"openai_features: {new_feature_json}")
            return new_feature_json
        else:
            diffs = diff_strings(json_to_check, new_feature_json)
            if diffs:
                logger.warning("diffs on second opinion:\n" + '\n'.join(diffs))
                if prefer_second_opinion:
                    logger.info(f"openai_features: {new_feature_json}")
                    return new_feature_json
            logger.info(f"openai_features: {json_to_check}")
            return json_to_check


def split_to_sentences(text: str) -> list[str]:
    return sent_tokenize(text)


if __name__ == '__main__':
    setup_logging(global_level="INFO")

    test_set = [
        "Hello, world!",
        "The Earth is round?",
        "This sucks!",
        "I hated broccoli as a kid but now I love it.",
        "Clifford is the big red dog.",
        "Let's go to 711.",
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
        ("In Revelation 13:18, it says, \"let the one who has understanding calculate the number of the beast, "
         "for it is the number of a man, and his number is 666\"."),
        "Maybe you should call 1-800-222-1222, the poison control hotline, or even 9-11!",
        "You can't break your 20 down the street because the 711 doesn't opens until 7am.",
    ]
    for test in test_set:
        print(f"""
Sentence: {test}
---
flair_text_feature_extraction: {flair_text_feature_extraction(test)}
openai_text_feature_extraction: {openai_text_feature_extraction(test, second_opinion=True)}
""")
