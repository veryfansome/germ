import logging

from bot.lang.classifiers import sentence_classifier

logger = logging.getLogger(__name__)


def match_reference_classification(test_sentence, reference_classification):
    new_classification = sentence_classifier.classify(test_sentence)
    logger.info(f"{test_sentence} {new_classification}")
    assert "functional_type" in new_classification
    assert new_classification["functional_type"] in reference_classification["functional_type"]

    assert "organizational_type" in new_classification
    assert new_classification["organizational_type"] in reference_classification["organizational_type"]

    assert "noticeable_emotions" in new_classification
    assert new_classification["noticeable_emotions"] in reference_classification["noticeable_emotions"]

    assert "reports_speech" in new_classification
    assert new_classification["reports_speech"] in reference_classification["reports_speech"]

    assert "uses_jargon" in new_classification
    assert new_classification["uses_jargon"] in reference_classification["uses_jargon"]


def test_sentence_classifier_case_declarative():
    """
    Simple declarative.

    :return:
    """
    match_reference_classification(
        "The cat sat on the mat.", {
            "functional_type": ["declarative"],
            "organizational_type": ["simple"],
            "noticeable_emotions": ["no"],
            "reports_speech": ["no"],
            "uses_jargon": ["no"],
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
            "noticeable_emotions": ["no"],
            "reports_speech": ["no"],
            "uses_jargon": ["no"],
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
            "noticeable_emotions": ["no"],
            "reports_speech": ["no"],
            "uses_jargon": ["no"],
        })
    match_reference_classification(
        "Should you need any help, feel free to ask.", {
            "functional_type": ["imperative"],
            "organizational_type": ["simple"],
            "noticeable_emotions": ["no"],
            "reports_speech": ["no"],
            "uses_jargon": ["no"],
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
            "noticeable_emotions": ["yes"],
            "reports_speech": ["no"],
            "uses_jargon": ["no"],
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
            "noticeable_emotions": ["no"],
            "reports_speech": ["no"],
            "uses_jargon": ["no"],
        })
    match_reference_classification(
        "Although it was raining, we decided to go for a walk.", {
            "functional_type": ["declarative"],
            "organizational_type": ["compound and/or complex"],
            "noticeable_emotions": ["no"],
            "reports_speech": ["no"],
            "uses_jargon": ["no"],
        })


def test_sentence_classifier_case_non_neutral_emotions():
    """
    Non-neutral emotions.

    :return:
    """
    match_reference_classification(
        "I am thrilled about the promotion!", {
            "functional_type": ["declarative"],
            "organizational_type": ["simple"],
            "noticeable_emotions": ["yes"],
            "reports_speech": ["no"],
            "uses_jargon": ["no"],
        })
    match_reference_classification(
        "He was devastated by the news.", {
            "functional_type": ["declarative"],
            "organizational_type": ["simple"],
            "noticeable_emotions": ["yes"],
            "reports_speech": ["no"],
            "uses_jargon": ["no"],
        })


def test_sentence_classifier_case_reports_speech():
    """
    Reports speech.

    :return:
    """
    match_reference_classification(
        "He said, 'I will be there soon.'", {
            "functional_type": ["declarative"],
            "organizational_type": ["simple"],
            "noticeable_emotions": ["no"],
            "reports_speech": ["yes"],
            "uses_jargon": ["no"],
        })
    match_reference_classification(
        "She asked if I had seen the movie.", {
            "functional_type": ["interrogative"],
            "organizational_type": ["simple"],
            "noticeable_emotions": ["no"],
            "reports_speech": ["yes"],
            "uses_jargon": ["no"],
        })


def test_sentence_classifier_case_jargon():
    """
    Jargon and slang.

    :return:
    """
    match_reference_classification(
        "The CPU is overheating due to overclocking.", {
            "functional_type": ["declarative"],
            "organizational_type": ["simple"],
            "noticeable_emotions": ["no"],
            "reports_speech": ["no"],
            "uses_jargon": ["yes"],
        })
    match_reference_classification(
        "He nailed the presentation with his killer pitch.", {
            "functional_type": ["declarative"],
            "organizational_type": ["simple"],
            "noticeable_emotions": ["yes"],
            "reports_speech": ["no"],
            "uses_jargon": ["yes"],
        })


if __name__ == "__main__":
    from observability.logging import setup_logging
    setup_logging()
