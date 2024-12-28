import logging

from bot.lang.classifiers import entity_classifier

logger = logging.getLogger(__name__)


def match_reference_entities(test_sentence, reference_classification):
    new_classification = entity_classifier.classify(test_sentence)
    logger.info(f"{test_sentence} {new_classification}")
    assert "entities" in new_classification
    for entity in new_classification["entities"]:
        assert "entity" in entity
        assert "entity_type" in entity
        assert "sentiment" in entity
        assert "semantic_roles" in entity
        signature_parts = [
            entity["entity"],
            entity["entity_type"],
            entity["sentiment"],
            *entity["semantic_roles"],
        ]
        check_signature = ", ".join(signature_parts)
        assert check_signature in reference_classification, check_signature


def test_entity_classifier_case_1():
    """
    Animal or Non-Humanoid Creature.

    :return:
    """
    match_reference_entities(
        "The elephant wandered through the savannah, searching for water.", [
            "elephant, animal or non-humanoid creature, neutral, theme",
            "savannah, geographical location or street address, neutral, location",
            "water, natural resource, artificial construction material, or other industrial input, neutral, goal",
        ])


def test_entity_classifier_case_2():
    """
    Building or Monument.

    :return:
    """
    match_reference_entities(
        "The Eiffel Tower is one of the most recognizable landmarks in Paris.", [
            "Eiffel Tower, building or monument, positive, theme",
            "Paris, geographical location or street address, positive, location",
        ])


def test_entity_classifier_case_3():
    """
    For-Profit Business Organization

    :return:
    """
    match_reference_entities(
        "Apple Inc. announced its latest product launch during the annual conference.", [
            "Apple Inc., for-profit business organization, neutral, agent",
            "annual conference, future event, neutral, location",
            "annual conference, future event, neutral, time",
            "latest product launch, future event, neutral, goal",
            "latest product launch, future event, neutral, theme",
        ])


def test_entity_classifier_case_4():
    """
    Currency.

    :return:
    """
    match_reference_entities(
        "The price of the new smartphone is listed at 999 euros.", [
            "999 euros, currency, neutral, quantity",
            "999 euros, currency, neutral, theme",
            "smartphone, computer, phone, book, or other recording, organization, or communication tool, neutral, theme",
        ])


def test_entity_classifier_case_5():
    """
    Clothing, Shoes, or Jewelry.

    :return:
    """
    match_reference_entities(
        "She wore a stunning diamond necklace to the gala.", [
            "diamond necklace, clothing, shoes, or jewelry, positive, theme",
            "gala, future event, neutral, location",  # Ignored the wore but I guess future events are possible
        ])


def test_entity_classifier_case_6():
    """
    Computer, Phone, Book, or Other Recording, Organization, or Communication Tool.

    :return:
    """
    match_reference_entities(
        "I read an interesting book on artificial intelligence last night.", [
            "artificial intelligence, scientific or technological idea, positive, theme",
            "book, computer, phone, book, or other recording, organization, or communication tool, positive, theme",
        ])
    match_reference_entities(
        "Right as the riot police began controlling the crowd, she shot off a tweet to her followers.", [
            "crowd, non-fictional person, neutral, patient",
            "crowd, social or cultural idea not related to economics, politics, or religion, neutral, patient",
            "riot police, government organization, neutral, agent",
            "tweet, computer, phone, book, or other recording, organization, or communication tool, neutral, instrument",
            "followers, non-fictional person, neutral, recipient",
            "followers, non-fictional person, positive, recipient",
        ])
    match_reference_entities(
        "She used an audio recorder to interview the cartel member about his connections and activities.", [
            "audio recorder, computer, phone, book, or other recording, organization, or communication tool, neutral, instrument",
            "cartel member, crime, terror, or paramilitary organization, negative, agent",
            "cartel member, crime, terror, or paramilitary organization, negative, patient",
            "cartel member, non-fictional person, negative, patient",
        ])


def test_entity_classifier_case_7():
    """
    Storage container.

    :return:
    """
    match_reference_entities(
        "She pulled out a pen and quickly scribbled the new information into her rolodex.", [
            "pen, utensil, instrument, machinery, or other mechanical tool, neutral, instrument",
            "rolodex, storage container, neutral, patient",
        ])
    match_reference_entities(
        "The box was filled with old photographs and letters.", [
            "box, storage container, neutral, theme",
            "photographs, non-fictional person, neutral, patient",
            "letters, non-fictional person, neutral, patient",
        ])


def test_entity_classifier_case_8():
    """
    Storage container.

    :return:
    """
    match_reference_entities(
        "The investigation revealed links to the notorious street gang.", [
            "notorious street gang, crime, terror, or paramilitary organization, negative, patient",
            "street gang, crime, terror, or paramilitary organization, negative, patient",
        ])


def test_entity_classifier_case_9():
    """
    Economic Idea.

    :return:
    """
    match_reference_entities(
        "I don't get how you can be in business without understanding basic supply and demand.", [
            "business, for-profit business organization, negative, theme",
            "supply and demand, economic idea, neutral, theme",
        ])


def test_entity_classifier_case_10():
    """
    Fictional characters.

    :return:
    """
    match_reference_entities(
        "Sherlock Holmes is known for his keen observation skills and his side-kick, Dr. Watson.", [
            "Dr. Watson, fictional character or imaginary persona, positive, accompaniment",
            "Sherlock Holmes, fictional character or imaginary persona, positive, agent",
        ])


def test_entity_classifier_case_11():
    """
    Fictional location.

    :return:
    """
    match_reference_entities(
        "Hogwarts School of Witchcraft and Wizardry is located in Scotland.", [
            "Hogwarts School of Witchcraft and Wizardry, building or monument, neutral, location",  # Not fictional?
            "Scotland, geographical location or street address, neutral, location",
        ])
    match_reference_entities(
        "Little did Dorthy know that she would, once again, be stuck in the Land of Oz.", [
            "Dorthy, fictional character or imaginary persona, neutral, experiencer",
            "Dorthy, non-fictional person, neutral, agent",
            "Dorthy, non-fictional person, neutral, experiencer",  # I guess we don't specify which Dorthy
            "Land of Oz, fictional location, neutral, location",
        ])


if __name__ == "__main__":
    from observability.logging import setup_logging
    setup_logging()

    #test_entity_classifier_case_1()
    #test_entity_classifier_case_2()
    #test_entity_classifier_case_3()
    #test_entity_classifier_case_4()
    #test_entity_classifier_case_5()
    #test_entity_classifier_case_6()
    #test_entity_classifier_case_7()
    #test_entity_classifier_case_8()
    #test_entity_classifier_case_9()
    #test_entity_classifier_case_10()
    test_entity_classifier_case_11()
