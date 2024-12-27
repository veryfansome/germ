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

    assert "change_in_state" in new_classification
    assert new_classification["change_in_state"] in reference_classification["change_in_state"]

    assert "noticeable_emotions" in new_classification
    assert new_classification["noticeable_emotions"] in reference_classification["noticeable_emotions"]

    assert "reports_speech" in new_classification
    assert new_classification["reports_speech"] in reference_classification["reports_speech"]

    assert "spatiality" in new_classification
    assert new_classification["spatiality"] in reference_classification["spatiality"]

    assert "temporality" in new_classification
    assert new_classification["temporality"] in reference_classification["temporality"]

    assert "uses_jargon" in new_classification
    assert new_classification["uses_jargon"] in reference_classification["uses_jargon"]


def test_sentence_classifier_case_1():
    """
    Simple declarative.

    :return:
    """
    match_reference_classification(
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


def test_sentence_classifier_case_2():
    """
    Interrogative.

    :return:
    """
    match_reference_classification(
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


def test_sentence_classifier_case_3():
    """
    Imperative.

    :return:
    """
    match_reference_classification(
        "Turn off the lights when you leave.", {
            "functional_type": ["imperative"],
            "organizational_type": ["simple"],
            "change_in_state": ["yes", "no"],
            "noticeable_emotions": ["no"],
            "reports_speech": ["no"],
            "spatiality": ["yes"],
            "temporality": ["yes", "no"],
            "uses_jargon": ["no"],
        })


def test_sentence_classifier_case_4():
    """
    Exclamatory.

    :return:
    """
    match_reference_classification(
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


def test_sentence_classifier_case_5():
    """
    Conditional.

    :return:
    """
    match_reference_classification(
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


def test_sentence_classifier_case_6():
    """
    Compound and/or complex.

    :return:
    """
    match_reference_classification(
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
    match_reference_classification(
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


def test_sentence_classifier_case_7():
    """
    Changes of state

    :return:
    """
    match_reference_classification(
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
    match_reference_classification(
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


def test_sentence_classifier_case_8():
    """
    Non-neutral emotions.

    :return:
    """
    match_reference_classification(
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
    match_reference_classification(
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


def test_sentence_classifier_case_9():
    """
    Reports speech.

    :return:
    """
    match_reference_classification(
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
    match_reference_classification(
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


def test_sentence_classifier_case_10():
    """
    Spatial changes

    :return:
    """
    match_reference_classification(
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
    match_reference_classification(
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


def test_sentence_classifier_case_11():
    """
    Temporal relationships

    :return:
    """
    match_reference_classification(
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
    match_reference_classification(
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
    match_reference_classification(
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


def test_sentence_classifier_case_12():
    """
    Jargon and slang.

    :return:
    """
    match_reference_classification(
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
    match_reference_classification(
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

    #test_sentence_classifier_case_1()
    #test_sentence_classifier_case_2()
    #test_sentence_classifier_case_3()
    #test_sentence_classifier_case_4()
    #test_sentence_classifier_case_5()
    #test_sentence_classifier_case_6()
    #test_sentence_classifier_case_7()
    #test_sentence_classifier_case_8()
    #test_sentence_classifier_case_9()
    #test_sentence_classifier_case_10()
    #test_sentence_classifier_case_11()
    test_sentence_classifier_case_12()
