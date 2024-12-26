import json
import logging

from bot.lang.utils import extract_openai_sentence_type_features

logger = logging.getLogger(__name__)


def match_reference_features(test_sentence, reference_features):
    extracted_features = json.loads(extract_openai_sentence_type_features(test_sentence))
    logger.info(f"{test_sentence} {extracted_features}")
    assert "functional_type" in extracted_features
    assert extracted_features["functional_type"] in reference_features["functional_type"]

    assert "organizational_type" in extracted_features
    assert extracted_features["organizational_type"] in reference_features["organizational_type"]

    assert "change_in_state" in extracted_features
    assert extracted_features["change_in_state"] in reference_features["change_in_state"]

    assert "noticeable_emotions" in extracted_features
    assert extracted_features["noticeable_emotions"] in reference_features["noticeable_emotions"]

    assert "reports_speech" in extracted_features
    assert extracted_features["reports_speech"] in reference_features["reports_speech"]

    assert "spatiality" in extracted_features
    assert extracted_features["spatiality"] in reference_features["spatiality"]

    assert "temporality" in extracted_features
    assert extracted_features["temporality"] in reference_features["temporality"]

    assert "uses_jargon" in extracted_features
    assert extracted_features["uses_jargon"] in reference_features["uses_jargon"]


def test_extract_openai_sentence_type_features_case_1():
    """
    Simple declarative.

    :return:
    """
    match_reference_features(
        "The cat sat on the mat.", {
            "functional_type": ["declarative"],
            "organizational_type": ["simple"],
            "change_in_state": ["no"],
            "noticeable_emotions": ["no"],
            "reports_speech": ["no"],
            "spatiality": ["yes"],
            "temporality": ["no"],
            "uses_jargon": ["no"],
        })


def test_extract_openai_sentence_type_features_case_2():
    """
    Interrogative.

    :return:
    """
    match_reference_features(
        "What time does the meeting start?", {
            "functional_type": ["interrogative"],
            "organizational_type": ["simple"],
            "change_in_state": ["no"],
            "noticeable_emotions": ["no"],
            "reports_speech": ["no"],
            "spatiality": ["no"],
            "temporality": ["yes"],
            "uses_jargon": ["no"],
        })


def test_extract_openai_sentence_type_features_case_3():
    """
    Imperative.

    :return:
    """
    match_reference_features(
        "Turn off the lights when you leave.", {
            "functional_type": ["imperative"],
            "organizational_type": ["simple"],
            "change_in_state": ["no"],
            "noticeable_emotions": ["no"],
            "reports_speech": ["no"],
            "spatiality": ["yes"],
            "temporality": ["yes", "no"],
            "uses_jargon": ["no"],
        })


def test_extract_openai_sentence_type_features_case_4():
    """
    Exclamatory.

    :return:
    """
    match_reference_features(
        "I can't believe we won the game!", {
            "functional_type": ["exclamatory"],
            "organizational_type": ["simple"],
            "change_in_state": ["no"],
            "noticeable_emotions": ["yes"],
            "reports_speech": ["no"],
            "spatiality": ["no"],
            "temporality": ["no"],
            "uses_jargon": ["no"],
        })


def test_extract_openai_sentence_type_features_case_5():
    """
    Conditional.

    :return:
    """
    match_reference_features(
        "Should you need any help, feel free to ask.", {
            "functional_type": ["conditional"],
            "organizational_type": ["simple"],
            "change_in_state": ["no"],
            "noticeable_emotions": ["no"],
            "reports_speech": ["no"],
            "spatiality": ["no"],
            "temporality": ["no"],
            "uses_jargon": ["no"],
        })


def test_extract_openai_sentence_type_features_case_6():
    """
    Compound and/or complex.

    :return:
    """
    match_reference_features(
        "She likes coffee, but he prefers tea.", {
            "functional_type": ["declarative"],
            "organizational_type": ["compound and/or complex"],
            "change_in_state": ["no"],
            "noticeable_emotions": ["no"],
            "reports_speech": ["no"],
            "spatiality": ["no"],
            "temporality": ["no"],
            "uses_jargon": ["no"],
        })
    match_reference_features(
        "Although it was raining, we decided to go for a walk.", {
            "functional_type": ["declarative"],
            "organizational_type": ["compound and/or complex"],
            "change_in_state": ["yes", "no"],
            "noticeable_emotions": ["no"],
            "reports_speech": ["no"],
            "spatiality": ["yes"],
            "temporality": ["yes"],
            "uses_jargon": ["no"],
        })


def test_extract_openai_sentence_type_features_case_7():
    """
    Changes of state

    :return:
    """
    match_reference_features(
        "The leaves turned from green to yellow.", {
            "functional_type": ["declarative"],
            "organizational_type": ["simple"],
            "change_in_state": ["yes"],
            "noticeable_emotions": ["no"],
            "reports_speech": ["no"],
            "spatiality": ["no"],
            "temporality": ["yes", "no"],
            "uses_jargon": ["no"],
        })
    match_reference_features(
        "He went from being a student to a teacher.", {
            "functional_type": ["declarative"],
            "organizational_type": ["simple"],
            "change_in_state": ["yes"],
            "noticeable_emotions": ["no"],
            "reports_speech": ["no"],
            "spatiality": ["no"],
            "temporality": ["yes", "no"],
            "uses_jargon": ["no"],
        })


def test_extract_openai_sentence_type_features_case_8():
    """
    Non-neutral emotions.

    :return:
    """
    match_reference_features(
        "I am thrilled about the promotion!", {
            "functional_type": ["exclamatory"],
            "organizational_type": ["simple"],
            "change_in_state": ["no"],
            "noticeable_emotions": ["yes"],
            "reports_speech": ["no"],
            "spatiality": ["no"],
            "temporality": ["no"],
            "uses_jargon": ["no"],
        })
    match_reference_features(
        "He was devastated by the news.", {
            "functional_type": ["declarative"],
            "organizational_type": ["simple"],
            "change_in_state": ["yes"],
            "noticeable_emotions": ["yes"],
            "reports_speech": ["no"],
            "spatiality": ["no"],
            "temporality": ["no"],
            "uses_jargon": ["no"],
        })


def test_extract_openai_sentence_type_features_case_9():
    """
    Reports speech.

    :return:
    """
    match_reference_features(
        "He said, 'I will be there soon.'", {
            "functional_type": ["declarative"],
            "organizational_type": ["simple"],
            "change_in_state": ["no"],
            "noticeable_emotions": ["no"],
            "reports_speech": ["yes"],
            "spatiality": ["yes", "no"],
            "temporality": ["yes", "no"],
            "uses_jargon": ["no"],
        })
    match_reference_features(
        "She asked if I had seen the movie.", {
            "functional_type": ["interrogative"],
            "organizational_type": ["simple"],
            "change_in_state": ["no"],
            "noticeable_emotions": ["no"],
            "reports_speech": ["yes"],
            "spatiality": ["no"],
            "temporality": ["no"],
            "uses_jargon": ["no"],
        })


def test_extract_openai_sentence_type_features_case_10():
    """
    Spatial changes

    :return:
    """
    match_reference_features(
        "The car moved from the garage to the driveway.", {
            "functional_type": ["declarative"],
            "organizational_type": ["simple"],
            "change_in_state": ["yes"],
            "noticeable_emotions": ["no"],
            "reports_speech": ["no"],
            "spatiality": ["yes"],
            "temporality": ["no"],
            "uses_jargon": ["no"],
        })
    match_reference_features(
        "He traveled from New York to Los Angeles.", {
            "functional_type": ["declarative"],
            "organizational_type": ["simple"],
            "change_in_state": ["yes", "no"],
            "noticeable_emotions": ["no"],
            "reports_speech": ["no"],
            "spatiality": ["yes"],
            "temporality": ["no"],
            "uses_jargon": ["no"],
        })


def test_extract_openai_sentence_type_features_case_11():
    """
    Temporal relationships

    :return:
    """
    match_reference_features(
        "He visited the museum last week.", {
            "functional_type": ["declarative"],
            "organizational_type": ["simple"],
            "change_in_state": ["yes", "no"],
            "noticeable_emotions": ["no"],
            "reports_speech": ["no"],
            "spatiality": ["yes", "no"],
            "temporality": ["yes"],
            "uses_jargon": ["no"],
        })
    match_reference_features(
        "She is studying for her exams.", {
            "functional_type": ["declarative"],
            "organizational_type": ["simple"],
            "change_in_state": ["no"],
            "noticeable_emotions": ["no"],
            "reports_speech": ["no"],
            "spatiality": ["no"],
            "temporality": ["yes", "no"],
            "uses_jargon": ["no"],
        })
    match_reference_features(
        "She plans to start a new job next month.", {
            "functional_type": ["declarative"],
            "organizational_type": ["simple"],
            "change_in_state": ["yes"],
            "noticeable_emotions": ["no"],
            "reports_speech": ["no"],
            "spatiality": ["no"],
            "temporality": ["yes"],
            "uses_jargon": ["no"],
        })


def test_extract_openai_sentence_type_features_case_12():
    """
    Jargon and slang.

    :return:
    """
    match_reference_features(
        "The CPU is overheating due to overclocking.", {
            "functional_type": ["declarative"],
            "organizational_type": ["simple"],
            "change_in_state": ["yes"],
            "noticeable_emotions": ["no"],
            "reports_speech": ["no"],
            "spatiality": ["no"],
            "temporality": ["no"],
            "uses_jargon": ["yes"],
        })
    match_reference_features(
        "He nailed the presentation with his killer pitch.", {
            "functional_type": ["declarative"],
            "organizational_type": ["simple"],
            "change_in_state": ["no"],
            "noticeable_emotions": ["yes"],
            "reports_speech": ["no"],
            "spatiality": ["no"],
            "temporality": ["no"],
            "uses_jargon": ["yes"],
        })


if __name__ == "__main__":
    from observability.logging import setup_logging
    setup_logging()

    #test_extract_openai_sentence_type_features_case_1()
    #test_extract_openai_sentence_type_features_case_2()
    #test_extract_openai_sentence_type_features_case_3()
    #test_extract_openai_sentence_type_features_case_4()
    #test_extract_openai_sentence_type_features_case_5()
    #test_extract_openai_sentence_type_features_case_6()
    #test_extract_openai_sentence_type_features_case_7()
    #test_extract_openai_sentence_type_features_case_8()
    #test_extract_openai_sentence_type_features_case_9()
    #test_extract_openai_sentence_type_features_case_10()
    #test_extract_openai_sentence_type_features_case_11()
    test_extract_openai_sentence_type_features_case_12()
