import json
import logging

from bot.lang.utils import extract_openai_sentence_type_features

logger = logging.getLogger(__name__)


def match_reference_features(test_sentence, reference_features):
    extracted_features = json.loads(extract_openai_sentence_type_features(test_sentence))
    logger.info(extracted_features)
    assert "change_of_state" in extracted_features
    assert extracted_features["change_of_state"] in reference_features["change_of_state"]
    assert "contains_interjection" in extracted_features
    assert extracted_features["contains_interjection"] in reference_features["contains_interjection"]
    assert "functional_type" in extracted_features
    assert extracted_features["functional_type"] in reference_features["functional_type"]
    assert "narrative_structure" in extracted_features
    assert extracted_features["narrative_structure"] in reference_features["narrative_structure"]
    assert "organizational_type" in extracted_features
    assert extracted_features["organizational_type"] in reference_features["organizational_type"]
    assert "spatiality" in extracted_features
    assert extracted_features["spatiality"] in reference_features["spatiality"]
    assert "style" in extracted_features
    assert extracted_features["style"] in reference_features["style"]
    assert "temporality" in extracted_features
    assert extracted_features["temporality"] in reference_features["temporality"]
    assert "voice" in extracted_features
    assert extracted_features["voice"] in reference_features["voice"]


def test_extract_openai_sentence_type_features_case_1():
    """
    Simple declarative.

    :return:
    """
    match_reference_features(
        "The cat sat on the mat.", {
            "change_of_state": ["static"],
            "contains_interjection": ["false"],
            "functional_type": ["declarative"],
            "narrative_structure": ["event-driven"],
            "organizational_type": ["simple"],
            "spatiality": ["static"],
            "style": ["formal"],
            "temporality": ["static present"],
            "voice": ["active"]})


def test_extract_openai_sentence_type_features_case_2():
    """
    Complex sentence with change of state.

    :return:
    """
    match_reference_features(
        "After the rain stopped, the flowers began to bloom.", {
            "change_of_state": ["complex changes", "subjects change"],
            "contains_interjection": ["false"],
            "functional_type": ["declarative"],
            "narrative_structure": ["event-driven"],
            "organizational_type": ["complex"],
            "spatiality": ["static", "subjects move"],
            "style": ["formal"],
            "temporality": ["past to present"],
            "voice": ["active"]})


def test_extract_openai_sentence_type_features_case_3():
    """
    Interrogatives.

    :return:
    """
    match_reference_features(
        "What time does the meeting start?", {
            "change_of_state": ["static"],
            "contains_interjection": ["false"],
            "functional_type": ["interrogative"],
            "narrative_structure": ["detail-driven"],
            "organizational_type": ["simple"],
            "spatiality": ["static"],
            "style": ["formal"],
            "temporality": ["static present"],
            "voice": ["active"]})


def test_extract_openai_sentence_type_features_case_4():
    """
    Exclamatory with interjection.

    :return:
    """
    match_reference_features(
        "Wow, that was an incredible performance!", {
            "change_of_state": ["static"],
            "contains_interjection": ["true"],
            "functional_type": ["exclamatory"],
            "narrative_structure": ["detail-driven"],
            "organizational_type": ["simple"],
            "spatiality": ["static"],
            "style": ["informal"],
            "temporality": ["static present"],
            "voice": ["active"]})


def test_extract_openai_sentence_type_features_case_5():
    """
    Imperative.

    :return:
    """
    match_reference_features(
        "Please close the door.", {
            "change_of_state": ["static"],
            "contains_interjection": ["false"],
            "functional_type": ["imperative"],
            "narrative_structure": ["event-driven"],
            "organizational_type": ["simple"],
            "spatiality": ["static"],
            "style": ["formal"],
            "temporality": ["static present"],
            "voice": ["active"]})


def test_extract_openai_sentence_type_features_case_6():
    """
    Conditional.

    :return:
    """
    match_reference_features(
        "If it rains tomorrow, we will cancel the picnic.", {
            "change_of_state": ["subjects change"],
            "contains_interjection": ["false"],
            "functional_type": ["conditional"],
            "narrative_structure": ["event-driven"],
            "organizational_type": ["complex"],
            "spatiality": ["static"],
            "style": ["formal"],
            "temporality": ["present to future"],
            "voice": ["active"]})


def test_extract_openai_sentence_type_features_case_7():
    """
    Compound sentence.

    :return:
    """
    match_reference_features(
        "I wanted to go for a walk, but it was too cold outside.", {
            "change_of_state": ["complex changes"],
            "contains_interjection": ["false"],
            "functional_type": ["declarative"],
            "narrative_structure": ["event-driven"],
            "organizational_type": ["compound"],
            "spatiality": ["static"],
            "style": ["formal"],
            "temporality": ["past to present"],
            "voice": ["active"]})


def test_extract_openai_sentence_type_features_case_8():
    """
    Complex-compound sentence.

    :return:
    """
    match_reference_features(
        "Although it was raining, we decided to go hiking, and we had a great time.", {
            "change_of_state": ["complex changes", "static"],
            "contains_interjection": ["false"],
            "functional_type": ["declarative"],
            "narrative_structure": ["event-driven"],
            "organizational_type": ["complex-compound"],
            "spatiality": ["subjects move"],
            "style": ["formal"],
            "temporality": ["past to present"],
            "voice": ["active"]})


def test_extract_openai_sentence_type_features_case_9():
    """
    Passive voice.

    :return:
    """
    match_reference_features(
        "The book was read by the entire class.", {
            "change_of_state": ["static"],
            "contains_interjection": ["false"],
            "functional_type": ["declarative"],
            "narrative_structure": ["event-driven"],
            "organizational_type": ["simple"],
            "spatiality": ["static"],
            "style": ["formal"],
            "temporality": ["static past"],
            "voice": ["passive"]})


def test_extract_openai_sentence_type_features_case_10():
    """
    Sentences with movement.

    :return:
    """
    match_reference_features(
        "The dancers twirled and leapt across the stage, their movements synchronized perfectly.", {
            "change_of_state": ["complex changes"],
            "contains_interjection": ["false"],
            "functional_type": ["declarative"],
            "narrative_structure": ["event-driven"],
            "organizational_type": ["compound"],
            "spatiality": ["complex movements"],
            "style": ["formal"],
            "temporality": ["static present"],
            "voice": ["active"]})
    match_reference_features(
        "The car sped down the highway, leaving a trail of dust behind.", {
            "change_of_state": ["complex changes"],
            "contains_interjection": ["false"],
            "functional_type": ["declarative"],
            "narrative_structure": ["event-driven"],
            "organizational_type": ["complex"],
            "spatiality": ["complex movements", "subjects move"],
            "style": ["formal"],
            "temporality": ["static present"],
            "voice": ["active"]})


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
    test_extract_openai_sentence_type_features_case_10()
