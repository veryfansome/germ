import logging

from bot.lang.classifiers import entity_classifier

logger = logging.getLogger(__name__)


def match_reference_entities(test_sentence, reference_classification):
    new_classification = entity_classifier.classify(test_sentence)
    logger.info(new_classification)


def test_entity_classifier_case_1():
    """
    Basic emotion.

    :return:
    """
    match_reference_entities(
        "The Renaissance was a pivotal period in European history.", {

        })


if __name__ == "__main__":
    from observability.logging import setup_logging
    setup_logging()

    test_entity_classifier_case_1()
