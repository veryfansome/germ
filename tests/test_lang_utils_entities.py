import logging

from bot.lang.utils import classify_emotions_using_openai

logger = logging.getLogger(__name__)


def match_reference_entities(test_sentence, reference_classification):
    new_classification = classify_emotions_using_openai(test_sentence)
    logger.info(new_classification)


#def test_extract_openai_emotion_features_case_1():
#    """
#    Basic emotion.
#
#    :return:
#    """
#    match_reference_emotions(
#        "I am so happy today!", {"happiness": {
#            "emotion_source": ["speaker"],
#            "emotion_source_entity_type": ["human", "person"],
#            "emotion_target": ["day", "today"],
#            "emotion_target_entity_type": ["time"],
#            "intensity": ["high"],
#            "nuance": ["simple"],
#            "synonymous_emotions": ["joy", "contentment", "delight"],
#            "opposite_emotions": ["sadness", "discontent"]}})


if __name__ == "__main__":
    from observability.logging import setup_logging
    setup_logging()

    #test_extract_openai_emotion_features_case_1()
