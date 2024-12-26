import logging

from bot.lang.utils import extract_openai_emotion_features

logger = logging.getLogger(__name__)


def have_common_element(list1, list2):
    for element in list1:
        if element in list2:
            return True
    return False


def match_reference_emotions(test_sentence, reference_emotions, match_all=True):
    extracted_features = extract_openai_emotion_features(test_sentence)
    logger.info(extracted_features)
    assert "emotions" in extracted_features
    assert len(extracted_features["emotions"]) >= len(reference_emotions)

    emotions_found = set()
    for extracted_emotion in extracted_features["emotions"]:
        emotions_found.add(extracted_emotion["emotion"])
        if not match_all and extracted_emotion["emotion"] not in reference_emotions:
            # Some emotions are flakey, so they may not be required for testing.
            continue

        # Since LLM outputs vary, we need to tolerate a range of answers.
        assert extracted_emotion["emotion_source"] in reference_emotions[extracted_emotion["emotion"]]["emotion_source"]
        assert extracted_emotion["emotion_source_entity_type"] in reference_emotions[extracted_emotion["emotion"]]["emotion_source_entity_type"]
        assert extracted_emotion["emotion_target"] in reference_emotions[extracted_emotion["emotion"]]["emotion_target"]
        assert extracted_emotion["emotion_target_entity_type"] in reference_emotions[extracted_emotion["emotion"]]["emotion_target_entity_type"]
        assert extracted_emotion["intensity"] in reference_emotions[extracted_emotion["emotion"]]["intensity"]
        assert extracted_emotion["nuance"] in reference_emotions[extracted_emotion["emotion"]]["nuance"]
        # Same ballpark is good.
        assert len(extracted_emotion["synonymous_emotions"]) >= 2
        assert have_common_element(extracted_emotion["synonymous_emotions"], reference_emotions[extracted_emotion["emotion"]]["synonymous_emotions"])
        assert len(extracted_emotion["opposite_emotions"]) >= 2
        assert have_common_element(extracted_emotion["opposite_emotions"], reference_emotions[extracted_emotion["emotion"]]["opposite_emotions"])
    if not match_all:
        # If not enforcing match all, make sure all enforced references are found.
        for emotion_name in reference_emotions:
            assert emotion_name in emotions_found


def test_extract_openai_emotion_features_case_1():
    """
    Basic emotion.

    :return:
    """
    match_reference_emotions(
        "I am so happy today!", {"happiness": {
            "emotion_source": ["speaker"],
            "emotion_source_entity_type": ["human", "person"],
            "emotion_target": ["day", "today"],
            "emotion_target_entity_type": ["time"],
            "intensity": ["high"],
            "nuance": ["simple"],
            "synonymous_emotions": ["joy", "contentment", "delight"],
            "opposite_emotions": ["sadness", "discontent"]}})
    match_reference_emotions(
        "He was angry when he saw the mess.", {"anger": {
            "emotion_source": ["He"],
            "emotion_source_entity_type": ["person"],
            "emotion_target": ["the mess"],
            "emotion_target_entity_type": ["situation"],
            "intensity": ["high"],
            "nuance": ["simple"],
            "synonymous_emotions": ["rage", "fury", "irritation"],
            "opposite_emotions": ["calm", "contentment"]}})


def test_extract_openai_emotion_features_case_2():
    """
    Complex emotion.

    :return:
    """
    match_reference_emotions(
        "Despite the chaos, she felt a strange sense of calm.", {"calm": {
            "emotion_source": ["she"],
            "emotion_source_entity_type": ["person"],
            "emotion_target": ["chaos"],
            "emotion_target_entity_type": ["situation"],
            "intensity": ["medium"],
            "nuance": ["complex"],
            "synonymous_emotions": ["serenity", "peace"],
            "opposite_emotions": ["anxiety", "tension"]}})
    match_reference_emotions(
        "He was both excited and nervous about the presentation.", {
            "excited": {
                "emotion_source": ["He"],
                "emotion_source_entity_type": ["person"],
                "emotion_target": ["the presentation"],
                "emotion_target_entity_type": ["event"],
                "intensity": ["medium"],
                "nuance": ["complex"],
                "synonymous_emotions": ["enthusiastic", "eager"],
                "opposite_emotions": ["bored", "disinterested"]
            },
            "nervous": {
                "emotion_source": ["He"],
                "emotion_source_entity_type": ["person"],
                "emotion_target": ["the presentation"],
                "emotion_target_entity_type": ["event"],
                "intensity": ["medium"],
                "nuance": ["complex"],
                "synonymous_emotions": ["anxious", "apprehensive"],
                "opposite_emotions": ["calm", "confident"]
            },
        })


def test_extract_openai_emotion_features_case_3():
    """
    Clear source and target.

    :return:
    """
    match_reference_emotions(
        "John was jealous of his brother's success.", {"jealousy": {
            "emotion_source": ["John"],
            "emotion_source_entity_type": ["person"],
            "emotion_target": ["brother's success"],
            "emotion_target_entity_type": ["achievement"],
            "intensity": ["medium"],
            "nuance": ["simple"],
            "synonymous_emotions": ["envy", "resentment"],
            "opposite_emotions": ["admiration", "contentment"]}})


def test_extract_openai_emotion_features_case_4():
    """
    Emotions with varying intensities.

    :return:
    """
    match_reference_emotions(
        "I am slightly annoyed by the noise.", {"annoyance": {
            "emotion_source": ["speaker"],
            "emotion_source_entity_type": ["human", "person"],
            "emotion_target": ["noise"],
            "emotion_target_entity_type": ["environment"],
            "intensity": ["low"],
            "nuance": ["simple"],
            "synonymous_emotions": ["irritation", "displeasure"],
            "opposite_emotions": ["calm", "contentment"]}})
    match_reference_emotions(
        "She was extremely furious when she found out.", {"furious": {
            "emotion_source": ["She"],
            "emotion_source_entity_type": ["person"],
            "emotion_target": ["the situation"],
            "emotion_target_entity_type": ["event", "situation"],
            "intensity": ["high"],
            "nuance": ["simple"],
            "synonymous_emotions": ["angry", "irate", "enraged"],
            "opposite_emotions": ["calm", "content", "pleased"]}})


def test_extract_openai_emotion_features_case_5():
    """
    Emotions with nuance.

    :return:
    """
    match_reference_emotions(
        "He felt a bittersweet nostalgia looking at old photos.", {
            "bittersweet": {
                "emotion_source": ["He"],
                "emotion_source_entity_type": ["person"],
                "emotion_target": ["old photos"],
                "emotion_target_entity_type": ["object"],
                "intensity": ["medium"],
                "nuance": ["complex"],
                "synonymous_emotions": ["mixed feelings", "poignant"],
                "opposite_emotions": ["joy", "joyful", "happiness", "happy"]
            },
            "nostalgia": {
                "emotion_source": ["He"],
                "emotion_source_entity_type": ["person"],
                "emotion_target": ["old photos"],
                "emotion_target_entity_type": ["object"],
                "intensity": ["medium"],
                "nuance": ["complex"],
                "synonymous_emotions": ["longing", "yearning"],
                "opposite_emotions": ["indifference", "apathy"]
            },
        })
    match_reference_emotions(
        "Her joy was tinged with a hint of regret.", {
            "joy": {
                "emotion_source": ["her"],
                "emotion_source_entity_type": ["person"],
                "emotion_target": ["her experience"],
                "emotion_target_entity_type": ["abstract", "emotion"],
                "intensity": ["medium"],
                "nuance": ["complex"],
                "synonymous_emotions": ["happiness", "delight"],
                "opposite_emotions": ["sadness", "regret"]
            },
            "regret": {
                "emotion_source": ["her"],
                "emotion_source_entity_type": ["person"],
                "emotion_target": ["her past actions"],
                "emotion_target_entity_type": ["abstract", "emotion", "event"],
                "intensity": ["medium"],
                "nuance": ["complex"],
                "synonymous_emotions": ["sorrow", "remorse"],
                "opposite_emotions": ["contentment", "satisfaction"]
            },
        })


def test_extract_openai_emotion_features_case_6():
    """
    Emotions in context.

    :return:
    """
    match_reference_emotions(
        "She was relieved when the test results came back negative.", {"relief": {
            "emotion_source": ["She"],
            "emotion_source_entity_type": ["person"],
            "emotion_target": ["test results"],
            "emotion_target_entity_type": ["event"],
            "intensity": ["medium", "high"],
            "nuance": ["simple"],
            "synonymous_emotions": ["comfort", "ease", "satisfaction"],
            "opposite_emotions": ["anxiety", "worry", "fear"]}})


def test_extract_openai_emotion_features_case_7():
    """
    Emotions in dialogue.

    :return:
    """
    match_reference_emotions(
        "‘I can't believe you did this,’ she said, her voice filled with disappointment.", {"disappointment": {
            "emotion_source": ["she"],
            "emotion_source_entity_type": ["person"],
            "emotion_target": ["the action"],
            "emotion_target_entity_type": ["event"],
            "intensity": ["high"],
            "nuance": ["simple"],
            "synonymous_emotions": ["dismay", "displeasure"],
            "opposite_emotions": ["satisfaction", "joy"]}})


def test_extract_openai_emotion_features_case_8():
    """
    Emotions in metaphorical language.

    :return:
    """
    match_reference_emotions(
        "Her heart was a storm of conflicting emotions.", {
            "conflicted": {
                "emotion_source": ["her", "her heart"],
                "emotion_source_entity_type": ["emotion", "emotional state", "metaphor", "person"],
                "emotion_target": ["emotions", "her heart", "her emotions", "her feelings"],
                "emotion_target_entity_type": ["abstract", "abstract concept", "emotion", "emotions", "metaphor"],
                "intensity": ["high"],
                "nuance": ["complex"],
                "synonymous_emotions": ["torn", "ambivalent", "uncertain"],
                "opposite_emotions": ["resolved", "certain", "clear"]
            },
        }, match_all=False)  # Stormy is sometimes included but not consistently.
    match_reference_emotions(
        "He felt like a leaf in the wind, lost and directionless.", {
            "confusion": {
                "emotion_source": ["He"],
                "emotion_source_entity_type": ["person"],
                "emotion_target": ["his situation"],
                "emotion_target_entity_type": ["abstract concept"],
                "intensity": ["high"],
                "nuance": ["complex"],
                "synonymous_emotions": ["disorientation", "bewilderment", "uncertain"],
                "opposite_emotions": ["clarity", "certainty"]
            },
            "vulnerability": {
                "emotion_source": ["He"],
                "emotion_source_entity_type": ["person"],
                "emotion_target": ["his state", "his state of being"],
                "emotion_target_entity_type": ["abstract concept"],
                "intensity": ["medium"],
                "nuance": ["complex"],
                "synonymous_emotions": ["fragility", "exposure"],
                "opposite_emotions": ["strength", "security"]
            },
        })


def test_extract_openai_emotion_features_case_9():
    """
    Emotions with ambiguity

    :return:
    """
    match_reference_emotions(
        "He smiled, but it was hard to tell if it was genuine.", {
            "uncertainty": {
                "emotion_source": ["he", "the observer"],
                "emotion_source_entity_type": ["person"],
                "emotion_target": ["the smile"],
                "emotion_target_entity_type": ["expression", "gesture"],
                "intensity": ["medium"],
                "nuance": ["complex"],
                "synonymous_emotions": ["doubt", "skepticism"],
                "opposite_emotions": ["confidence", "trust"]
            },
        }, match_all=False)  # Curiosity appears sometimes
    match_reference_emotions(
        "Her expression was unreadable, a mix of emotions.", {
            "confusion": {
                "emotion_source": ["her"],
                "emotion_source_entity_type": ["person"],
                "emotion_target": ["her expression"],
                "emotion_target_entity_type": ["expression"],
                "intensity": ["medium"],
                "nuance": ["complex"],
                "synonymous_emotions": ["uncertainty", "bewilderment"],
                "opposite_emotions": ["clarity", "certainty"]
            },
        }, match_all=False)  # Intrigue, curiosity, anxiety, ambivalence app appear sometimes.


if __name__ == "__main__":
    from observability.logging import setup_logging
    setup_logging()

    #test_extract_openai_emotion_features_case_1()
    #test_extract_openai_emotion_features_case_2()
    #test_extract_openai_emotion_features_case_3()
    #test_extract_openai_emotion_features_case_4()
    #test_extract_openai_emotion_features_case_5()
    #test_extract_openai_emotion_features_case_6()
    #test_extract_openai_emotion_features_case_7()
    #test_extract_openai_emotion_features_case_8()
    test_extract_openai_emotion_features_case_9()
