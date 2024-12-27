import logging

from bot.lang.utils import equivalence_classifier

logger = logging.getLogger(__name__)


def match_reference_equivalences(test_sentence, reference_classification, review: bool = False):
    new_classification = equivalence_classifier.classify(test_sentence, review=review)
    logger.info(f"{test_sentence} {new_classification}")
    assert "equivalences" in new_classification
    for equivalence in new_classification["equivalences"]:
        assert "X" in equivalence
        assert "Y" in equivalence
        assert "relationship_type" in equivalence
        signature_parts = [
            equivalence["X"],
            equivalence["relationship_type"],
            equivalence["Y"],
        ]
        if "conditions" in equivalence:
            signature_parts += equivalence["conditions"]
        check_signature = ", ".join(signature_parts)
        assert check_signature in reference_classification, check_signature


def test_equivalence_classifier_case_1():
    """
    Identity.

    :return:
    """
    match_reference_equivalences("The capital of France is Paris.", [
        "capital of France, identical, Paris",
        "The capital of France, identical, Paris",
    ])
    match_reference_equivalences("Water is H2O.", [
        "Water, identical, H2O",
        "water, identical, H2O",
    ])


def test_equivalence_classifier_case_2():
    """
    Equivalence.

    :return:
    """
    match_reference_equivalences("A dollar is equivalent to 100 cents.", [
        "1 dollar, equivalent, 100 cents",
        "a dollar, equivalent, 100 cents",
        "A dollar, equivalent, 100 cents",
    ])
    match_reference_equivalences("The freezing point of water is 0 degrees Celsius.", [
        "freezing point of water, equivalent, 0 degrees Celsius",
    ])


def test_equivalence_classifier_case_3():
    """
    Synonym but can be on the border.

    :return:
    """
    match_reference_equivalences("A physician is a doctor.", [
        "physician, synonym, doctor",
    ])
    match_reference_equivalences("To commence is to begin.", [
        "to commence, synonym, to begin",
    ])


def test_equivalence_classifier_case_4():
    """
    Subset.

    :return:
    """
    match_reference_equivalences("A square is a rectangle.", [
        "square, subset, rectangle",
    ])
    match_reference_equivalences("All cats are felines.", [
        "cats, subset, felines",
    ])


def test_equivalence_classifier_case_5():
    """
    Definition but some on the border.

    :return:
    """
    match_reference_equivalences("The boiling point of water is 100 degrees Celsius at standard atmospheric pressure.", [
        "boiling point of water, equivalent, 100 degrees Celsius",
        "boiling point of water, definition, 100 degrees Celsius, at standard atmospheric pressure",
        "boiling point of water, equivalent, 100 degrees Celsius, at standard atmospheric pressure",
        "boiling point of water, definition, 100 degrees Celsius at standard atmospheric pressure",
    ])
    match_reference_equivalences("The area of a circle can be expressed as π times the radius squared.", [
        "area of a circle, definition, π times the radius squared",
        "area of a circle, equivalent, π times the radius squared",
    ])
    match_reference_equivalences("A triangle is a three-sided polygon.", [
        "triangle, definition, three-sided polygon",
    ])


if __name__ == "__main__":
    from observability.logging import setup_logging
    setup_logging()

    #test_equivalence_classifier_case_1()
    #test_equivalence_classifier_case_2()
    #test_equivalence_classifier_case_3()
    test_equivalence_classifier_case_4()
    #test_equivalence_classifier_case_5()
