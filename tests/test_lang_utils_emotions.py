import logging

from bot.lang.utils import emotion_to_entity_classifier

logger = logging.getLogger(__name__)


def have_common_element(list1, list2):
    for element in list1:
        if element in list2:
            return True
    return False


def match_reference_emotions(test_sentence, emotion_map, reference_classification, match_all=True):
    new_classification = emotion_to_entity_classifier.classify(test_sentence)
    logger.info(new_classification)
    assert "emotions" in new_classification
    assert len(new_classification["emotions"]) >= len(reference_classification)

    emotions_found = set()
    for emotion in new_classification["emotions"]:
        emotions_found.add(emotion["emotion"])
        if not match_all and emotion["emotion"] not in emotion_map:
            # Some emotions are flakey, so they may not be required for testing.
            continue

        # Since LLM outputs vary, we need to tolerate a range of answers. Same ballpark is good.
        ref_key = emotion_map[emotion["emotion"]]
        emotions_found.add(ref_key)
        assert emotion["felt_by"] in reference_classification[ref_key]["felt_by"], emotion["emotion"]
        assert emotion["felt_by_entity_type"] in reference_classification[ref_key]["felt_by_entity_type"], emotion["emotion"]
        assert emotion["felt_towards"] in reference_classification[ref_key]["felt_towards"], emotion["emotion"]
        assert emotion["felt_towards_entity_type"] in reference_classification[ref_key]["felt_towards_entity_type"], emotion["emotion"]
    if not match_all:
        # If not enforcing match all, make sure all enforced references are found.
        for emotion_name in reference_classification:
            assert emotion_name in emotions_found


def test_emotion_to_entity_classifier_case_1():
    """
    Basic emotion.

    :return:
    """
    match_reference_emotions(
        "I am so happy today!",
        {"happiness": "happy", "happy": "happy"},
        {
            "happy": {
                "felt_by": ["speaker", "the speaker", "user"],
                "felt_by_entity_type": ["human", "person"],
                "felt_towards": ["day", "the day", "today", "", None],
                "felt_towards_entity_type": ["time", "", None],
            }
        })
    match_reference_emotions(
        "He was angry when he saw the mess.",
        {"anger": "anger", "angry": "anger"},
        {
            "anger": {
                "felt_by": ["He"],
                "felt_by_entity_type": ["person"],
                "felt_towards": ["the mess"],
                "felt_towards_entity_type": ["object", "situation"],
            }
        })


def test_emotion_to_entity_classifier_case_2():
    """
    Complex emotion.

    :return:
    """
    match_reference_emotions(
        "Despite the chaos, she felt a strange sense of calm.",
        {"calm": "calm"},
        {
            "calm": {
                "felt_by": ["she"],
                "felt_by_entity_type": ["person"],
                "felt_towards": ["chaos"],
                "felt_towards_entity_type": ["situation"],
            }
        })
    match_reference_emotions(
        "He was both excited and nervous about the presentation.",
        {"excited": "excited", "nervous": "nervous"},
        {
            "excited": {
                "felt_by": ["He"],
                "felt_by_entity_type": ["person"],
                "felt_towards": ["presentation", "the presentation"],
                "felt_towards_entity_type": ["event"],
            },
            "nervous": {
                "felt_by": ["He"],
                "felt_by_entity_type": ["person"],
                "felt_towards": ["presentation", "the presentation"],
                "felt_towards_entity_type": ["event"],
            },
        })


def test_emotion_to_entity_classifier_case_3():
    """
    Clear source and target.

    :return:
    """
    match_reference_emotions(
        "John was jealous of his brother's success.",
        {"jealousy": "jealousy"},
        {
            "jealousy": {
                "felt_by": ["John"],
                "felt_by_entity_type": ["person"],
                "felt_towards": ["brother's success", "his brother's success"],
                "felt_towards_entity_type": ["achievement", "person", "success"],
            }
        })


def test_emotion_to_entity_classifier_case_4():
    """
    Emotions with varying intensities.

    :return:
    """
    match_reference_emotions(
        "I am slightly annoyed by the noise.",
        {"annoyance": "annoyance", "annoyed": "annoyance"},
        {
            "annoyance": {
                "felt_by": ["speaker", "the speaker"],
                "felt_by_entity_type": ["human", "person"],
                "felt_towards": ["noise", "the noise"],
                "felt_towards_entity_type": ["environment", "sound"],
            }
        })
    match_reference_emotions(
        "She was extremely furious when she found out.",
        {"furious": "furious"},
        {
            "furious": {
                "felt_by": ["She", "she"],
                "felt_by_entity_type": ["person"],
                "felt_towards": [
                    "the situation",
                    "the situation of finding out something",
                    "the situation of finding out something unexpected",
                ],
                "felt_towards_entity_type": ["event", "situation"],
            }
        })


def test_emotion_to_entity_classifier_case_5():
    """
    Emotions with nuance.

    :return:
    """
    match_reference_emotions(
        "He felt a bittersweet nostalgia looking at old photos.",
        {
            "bittersweet": "bittersweet nostalgia",
            "bittersweet nostalgia": "bittersweet nostalgia",
            "nostalgia": "bittersweet nostalgia"
        },
        {
            "bittersweet nostalgia": {
                "felt_by": ["He"],
                "felt_by_entity_type": ["person"],
                "felt_towards": ["old photos"],
                "felt_towards_entity_type": ["object"],
            },
        })
    match_reference_emotions(
        "Her joy was tinged with a hint of regret.",
        {"joy": "joy", "regret": "regret"},
        {
            "joy": {
                "felt_by": ["Her", "her"],
                "felt_by_entity_type": ["person"],
                "felt_towards": ["joy", "regret", "the moment"],
                "felt_towards_entity_type": ["abstract concept", "emotion", "event", "situation"],
            },
            "regret": {
                "felt_by": ["Her", "her"],
                "felt_by_entity_type": ["person"],
                "felt_towards": ["joy", "the moment", "the moment or past actions", "the past or a decision made",
                                 "the past or decision made", "the situation", "the situation or past actions", None],
                "felt_towards_entity_type": ["abstract concept", "concept", "emotion", "event", "situation", None],
            },
        })


def test_emotion_to_entity_classifier_case_6():
    """
    Emotions in context.

    :return:
    """
    match_reference_emotions(
        "She was relieved when the test results came back negative.",
        {"relief": "relief"},
        {
            "relief": {
                "felt_by": ["She", "she"],
                "felt_by_entity_type": ["person"],
                "felt_towards": ["test results"],
                "felt_towards_entity_type": ["event", "outcome"],
            }
        })


def test_emotion_to_entity_classifier_case_7():
    """
    Emotions in dialogue.

    :return:
    """
    match_reference_emotions(
        "‘I can't believe you did this,’ she said, her voice filled with disappointment.",
        {"disappointment": "disappointment"},
        {
            "disappointment": {
                "felt_by": ["she"],
                "felt_by_entity_type": ["person"],
                "felt_towards": ["the action", "the action of the other person", "you"],
                "felt_towards_entity_type": ["action", "person"],
            }
        })


def test_emotion_to_entity_classifier_case_8():
    """
    Emotions in metaphorical language.

    :return:
    """
    match_reference_emotions(
        "Her heart was a storm of conflicting emotions.",
        {"conflicted": "conflicted"},
        {
            "conflicted": {
                "felt_by": ["her", "her heart"],
                "felt_by_entity_type": ["person"],
                "felt_towards": ["emotions", "her own emotions", "her own feelings"],
                "felt_towards_entity_type": ["abstract", "abstract concept", "abstract emotion", "emotions"],
            },
        }, match_all=False)  # Stormy is sometimes included but not consistently.
    match_reference_emotions(
        "He felt like a leaf in the wind, lost and directionless.",
        {
            "confusion": "lost",
            "directionless": "lost",
            "lost": "lost",
            "vulnerability": "vulnerability"},
        {
            "vulnerability": {
                "felt_by": ["He"],
                "felt_by_entity_type": ["person"],
                "felt_towards": ["his life", "his state", "his state of being"],
                "felt_towards_entity_type": ["abstract concept", "situation"],
            },
            "lost": {
                "felt_by": ["He"],
                "felt_by_entity_type": ["person"],
                "felt_towards": ["his life", "his life path", "his situation"],
                "felt_towards_entity_type": ["abstract concept", "concept", "life", "life path", "situation", "state"],
            },
        })


def test_emotion_to_entity_classifier_case_9():
    """
    Emotions with ambiguity

    :return:
    """
    match_reference_emotions(
        "He smiled, but it was hard to tell if it was genuine.",
        {"uncertainty": "uncertainty"},
        {
            "uncertainty": {
                "felt_by": ["he", "the observer"],
                "felt_by_entity_type": ["person"],
                "felt_towards": ["his smile", "the smile"],
                "felt_towards_entity_type": ["expression", "facial expression", "gesture"],
            },
        }, match_all=False)  # Curiosity appears sometimes
    match_reference_emotions(
        "Her expression was unreadable, a mix of emotions.",
        {"confusion": "confusion"},
        {
            "confusion": {
                "felt_by": ["her"],
                "felt_by_entity_type": ["person"],
                "felt_towards": ["her expression", "the situation"],
                "felt_towards_entity_type": ["expression", "facial expression", "situation"],
            },
        }, match_all=False)  # Intrigue, curiosity, anxiety, ambivalence app appear sometimes.


if __name__ == "__main__":
    from observability.logging import setup_logging
    setup_logging()

    #test_emotion_to_entity_classifier_case_1()
    #test_emotion_to_entity_classifier_case_2()
    #test_emotion_to_entity_classifier_case_3()
    #test_emotion_to_entity_classifier_case_4()
    #test_emotion_to_entity_classifier_case_5()
    #test_emotion_to_entity_classifier_case_6()
    #test_emotion_to_entity_classifier_case_7()
    #test_emotion_to_entity_classifier_case_8()
    test_emotion_to_entity_classifier_case_9()
