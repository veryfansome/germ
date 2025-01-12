import logging

from bot.lang.classifiers import get_sentence_classifier

logger = logging.getLogger(__name__)


def match_reference_classification(test_sentence, reference_classification):
    new_classification = get_sentence_classifier({
        "contains_code": {
            "type": "string",
            "description": "Does the sentence include log or error messages or snippets of computer code?",
            "enum": ["yes", "no"],
        },
        "contains_emotion": {
            "type": "string",
            "description": "Does the sentence include non-neutral emotions?",
            "enum": ["yes", "no"],
        },
        "contains_speech": {
            "type": "string",
            "description": "Does the sentence report direct or paraphrased speech?",
            "enum": ["yes", "no"],
        },
    }).classify(test_sentence)
    logger.info(f"{test_sentence} {new_classification}")
    assert "functional_type" in new_classification
    assert new_classification["functional_type"] in reference_classification["functional_type"]

    assert "organizational_type" in new_classification
    assert new_classification["organizational_type"] in reference_classification["organizational_type"]

    assert "contains_code" in new_classification
    assert new_classification["contains_code"] in reference_classification["contains_code"]

    assert "contains_emotion" in new_classification
    assert new_classification["contains_emotion"] in reference_classification["contains_emotion"]

    assert "contains_speech" in new_classification
    assert new_classification["contains_speech"] in reference_classification["contains_speech"]


def test_sentence_classifier_case_declarative():
    """
    Simple declarative.

    :return:
    """
    match_reference_classification(
        "The cat sat on the mat.", {
            "functional_type": ["declarative"],
            "organizational_type": ["simple"],
            "contains_code": ["no"],
            "contains_emotion": ["no"],
            "contains_speech": ["no"],
        })


def test_sentence_classifier_case_interrogative():
    """
    Interrogative.

    :return:
    """
    match_reference_classification(
        "What time does the meeting start?", {
            "functional_type": ["interrogative"],
            "organizational_type": ["simple"],
            "contains_code": ["no"],
            "contains_emotion": ["no"],
            "contains_speech": ["no"],
        })


def test_sentence_classifier_case_imperative():
    """
    Imperative.

    :return:
    """
    match_reference_classification(
        "Turn off the lights when you leave.", {
            "functional_type": ["imperative"],
            "organizational_type": ["simple"],
            "contains_code": ["no"],
            "contains_emotion": ["no"],
            "contains_speech": ["no"],
        })
    match_reference_classification(
        "Should you need any help, feel free to ask.", {
            "functional_type": ["imperative"],
            "organizational_type": ["simple"],
            "contains_code": ["no"],
            "contains_emotion": ["no"],
            "contains_speech": ["no"],
        })


def test_sentence_classifier_case_exclamatory():
    """
    Exclamatory.

    :return:
    """
    match_reference_classification(
        "I can't believe we won the game!", {
            "functional_type": ["declarative"],
            "organizational_type": ["simple"],
            "contains_code": ["no"],
            "contains_emotion": ["yes"],
            "contains_speech": ["no"],
        })


def test_sentence_classifier_case_compound_and_or_complex():
    """
    Compound and/or complex.

    :return:
    """
    match_reference_classification(
        "She likes coffee, but he prefers tea.", {
            "functional_type": ["declarative"],
            "organizational_type": ["compound and/or complex"],
            "contains_code": ["no"],
            "contains_emotion": ["no"],
            "contains_speech": ["no"],
        })
    match_reference_classification(
        "Although it was raining, we decided to go for a walk.", {
            "functional_type": ["declarative"],
            "organizational_type": ["compound and/or complex"],
            "contains_code": ["no"],
            "contains_emotion": ["no"],
            "contains_speech": ["no"],
        })


def test_sentence_classifier_case_contains_code():
    """
    Contains code

    :return:
    """
    match_reference_classification(
        "I tried your suggestion but I got an error, `RuntimeError: Failed to open database`.", {
            "functional_type": ["declarative"],
            "organizational_type": ["simple"],
            "contains_code": ["yes"],
            "contains_emotion": ["no"],
            "contains_speech": ["no"],
        })


def test_sentence_classifier_case_contains_emotion():
    """
    Non-neutral emotions.

    :return:
    """
    match_reference_classification(
        "I am thrilled about the promotion!", {
            "functional_type": ["declarative"],
            "organizational_type": ["simple"],
            "contains_code": ["no"],
            "contains_emotion": ["yes"],
            "contains_speech": ["no"],
        })
    match_reference_classification(
        "He was devastated by the news.", {
            "functional_type": ["declarative"],
            "organizational_type": ["simple"],
            "contains_code": ["no"],
            "contains_emotion": ["yes"],
            "contains_speech": ["no"],
        })


def test_sentence_classifier_case_contains_speech():
    """
    Reports speech.

    :return:
    """
    match_reference_classification(
        "He said, 'I will be there soon.'", {
            "functional_type": ["declarative"],
            "organizational_type": ["simple"],
            "contains_code": ["no"],
            "contains_emotion": ["no"],
            "contains_speech": ["yes"],
        })
    match_reference_classification(
        "She asked if I had seen the movie.", {
            "functional_type": ["interrogative"],
            "organizational_type": ["simple"],
            "contains_code": ["no"],
            "contains_emotion": ["no"],
            "contains_speech": ["yes"],
        })


if __name__ == "__main__":
    from observability.logging import setup_logging
    setup_logging()

    test_sentence_classifier_case_contains_code()
