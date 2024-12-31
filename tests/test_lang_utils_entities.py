import logging
from prometheus_client import Counter

from bot.lang.classifiers import get_entity_type_classifier

logger = logging.getLogger(__name__)
entity_counter = Counter('entity_count', 'Total number of classifications.', ["entity", "entity_type"])


def match_reference_entities(test_sentence, reference_classification):
    new_classification = get_entity_type_classifier().classify(test_sentence)
    logger.info(f"{test_sentence} {new_classification}")
    assert "entities" in new_classification
    for entity in new_classification["entities"]:
        assert "entity" in entity
        assert "entity_type" in entity
        entity_counter.labels(entity=entity["entity"], entity_type=["entity_type"]).inc()
        signature_parts = [
            str(entity["entity"]),
            str(entity["entity_type"]),
        ]
        check_signature = ", ".join(signature_parts)
        assert check_signature in reference_classification, check_signature


def test_entity_classifier_case_abstract_ability_or_attribute():
    """
    Abstract ability or attribute

    :return:
    """
    match_reference_entities(
        "Her kindness is truly admirable and inspiring.", [
            "admiration, concept",
            "inspiration, concept",
            "inspiring, concept",
            "inspiring, trait",
            "kindness, trait",
        ])
    match_reference_entities(
        "Intelligence is often measured by one's ability to solve complex problems.", [
            "Intelligence, concept",
            "complex problems, concept"
        ])


def test_entity_classifier_case_animal_or_non_humanoid():
    """
    Animal or Non-Humanoid Creature.

    :return:
    """
    match_reference_entities(
        "The elephant wandered through the savannah, searching for water.", [
            "elephant, fauna",
            "savannah, terrain",
            "water, material or substance",
        ])
    match_reference_entities(
        "In the dense jungle, a jaguar prowled silently through the underbrush.", [
            "jaguar, fauna",
            "jungle, area",
            "jungle, terrain",
            "underbrush, terrain",
        ])
    match_reference_entities(
        "Dolphins are known for their intelligence and playful behavior.", [
            "Dolphins, fauna",
            "intelligence, concept",
            "intelligence, trait",
            "playful behavior, trait",
        ])


def test_entity_classifier_case_article_book_document_or_other_text():
    """
    Article, book, document, post or other textual artifact.

    :return:
    """
    match_reference_entities(
        "I read an interesting book on artificial intelligence last night.", [
            "artificial intelligence, concept",
            "book, container",
            "book, media",
        ])
    match_reference_entities(
        "The research paper provided new insights into climate change.", [
            "climate change, concept",
            "research paper, media",
        ])
    match_reference_entities(
        "I just finished reading the assigned chapter from \"To Kill a Mockingbird\".", [
            "To Kill a Mockingbird, media",
            "chapter, concept",
        ])


def test_entity_classifier_case_artistic_or_literary_concept():
    """
    Artistic or literary concept

    :return:
    """
    match_reference_entities(
        "Surrealism challenges the boundaries of reality and imagination.", [
            "Surrealism, concept",
            "imagination, concept",
            "reality, concept",
        ])
    match_reference_entities(
        "The works in this period of the painter's life is filled with remorse and repentance related imagery.", [
            "imagery, concept",
            "imagery, media",
            "painter, person",
            "period of the painter's life, duration",
            "remorse and repentance related imagery, concept",
            "remorse, concept",
            "repentance, concept",
        ])


def test_entity_classifier_case_audio_image_video_or_other_media():
    """
    Audio, image, video, or other captured or recorded media artifact

    :return:
    """
    match_reference_entities(
        ("She used an audio recorder to tape the interview with the cartel member, "
         "after being taken to an unspecified location."), [
            "audio recorder, device",
            "cartel member, person",
            "interview, activity",
            "interview, event",
            "unspecified location, place",
        ])
    match_reference_entities(
        "The podcast episode explored the history of jazz music.", [
            "history, concept",
            "jazz music, concept",
            "jazz music, media",
            "podcast episode, media",
        ])
    match_reference_entities(
        "A viral video of a cat playing the piano captivated millions online.", [
            "cat, fauna",
            "millions, quantity",
            "piano, instrument",
            "video, media",
            "viral video, media",
        ])


def test_entity_classifier_case_clothing_shoes_or_jewelry():
    """
    Clothing, Shoes, or Jewelry.

    :return:
    """
    match_reference_entities(
        "She wore a stunning diamond necklace to the gala.", [
            "diamond necklace, apparel",
            "gala, event",
        ])
    match_reference_entities(
        "His wore designer Air Jordans to the game.", [
            "Air Jordans, apparel",
            "game, event",
        ])
    match_reference_entities(
        "His Chuck Taylors have seen better days.", [
            "Chuck Taylors, apparel",
        ])


def test_entity_classifier_case_comment_message_letter_or_communication_artifact():
    """
    Comment, message, letter, or other communication artifact

    :return:
    """
    match_reference_entities(
        "Right before she disappeared, she sent me three messages.", [
            "messages, media",
            "she, person",
            "three messages, event",
            "three messages, media",
            "three messages, quantity",
        ])
    match_reference_entities(
        "The email contained important updates about the project.", [
            "email, concept",
            "email, media",
            "project, concept",
            "project, initiative or objective",
            "updates, activity",
            "updates, concept",
            "updates, event",
        ])
    match_reference_entities(
        "Right as the riot police began controlling the crowd, she shot off a tweet to her followers.", [
            "crowd, concept",
            "crowd, phenomenon",
            "followers, person",
            "riot police, organization",
            "tweet, activity",
            "tweet, media",
        ])


def test_entity_classifier_case_computer_phone_or_electronic_device():
    """
    Computer, phone, or personal electronic device.

    :return:
    """
    match_reference_entities(
        "He upgraded to the latest smartphone model.", [
            "smartphone model, device",
        ])
    match_reference_entities(
        "His laptop fell down three flights of stairs and crashed against a concrete surface.", [
            "concrete surface, material or substance",
            "concrete surface, structure",
            "flights of stairs, structure",
            "laptop, device",
        ])
    match_reference_entities(
        "He had been very excited about his Google glasses but he stopped wearing them after the backlash.", [
            "Google glasses, device",
            "backlash, concept",
            "backlash, phenomenon",
        ])


def test_entity_classifier_case_construction_or_industrial_input():
    """
    Construction or industrial input.

    :return:
    """
    match_reference_entities(
        "The building was constructed using high-quality steel beams.", [
            "building, structure",
            "steel beams, material or substance",
        ])
    match_reference_entities(
        "Cement is as important today as it was to the Romans.", [
            "Cement, material or substance",
            "Romans, person",
        ])
    match_reference_entities(
        "The renewed interest in nuclear power due to AI has sparked a rally in palladium prices.", [
            "AI, concept",
            "nuclear power, concept",
            "palladium prices, currency",
        ])


def test_entity_classifier_case_city_county_and_other_localities():
    """
    Cities and other localities.

    :return:
    """
    match_reference_entities(
        "He used to live in Alameda County but moved further south along the 880.", [
            "880, road",
            "Alameda County, area",
        ])
    match_reference_entities(
        "His family has a pig farm in Yorkshire, surrounded by stunning landscape.", [
            "Yorkshire, area",
            "Yorkshire, place",
            "landscape, phenomenon",
            "pig farm, activity",
            "pig farm, area",
            "pig farm, organization",
            "pig farm, structure",
            "stunning landscape, area",
            "stunning landscape, phenomenon",
            "stunning landscape, terrain",
        ])
    match_reference_entities(
        "In Quebec, they speak French.", [
            "French, language",
            "Quebec, place",
        ])
    match_reference_entities(
        ("The staged explosion on the South Manchurian Railway in Liaoning Province was, famously, the pretext for "
         "the invasion of Manchuria by Imperial Japan."), [
            "Imperial Japan, organization",
            "Liaoning Province, area",
            "Liaoning Province, place",
            "Manchuria, area",
            "South Manchurian Railway, structure",
            "invasion, event",
            "staged explosion, event",
        ])
    match_reference_entities(
        ("On June 30, 1908, in a remote area near the Tunguska River in Siberia, a massive blast flattened "
         "an estimated 2,000 square kilometers (about 770 square miles) of forest, knocking down "
         "around 80 million trees."), [
            "2,000 square kilometers, measurement",
            "770 square miles, measurement",
            "80 million trees, quantity",
            "June 30, 1908, date or time",
            "Siberia, area",
            "Tunguska River, place",
        ])


def test_entity_classifier_case_crime_terror_or_paramilitary():
    """
    Crime, terror, or paramilitary organization.

    :return:
    """
    match_reference_entities(
        "The investigation revealed links to the notorious street gang.", [
            "investigation, activity",
            "investigation, event",
            "street gang, organization",
        ])
    match_reference_entities(
        "Having pushed Asad out of Syria, Hayat Tahrir al-Sham now must form a new legitimate government.", [
            "Asad, person",
            "Hayat Tahrir al-Sham, organization",
            "Syria, place",
            "government, concept",
        ])


def test_entity_classifier_case_currency():
    """
    Currency.

    :return:
    """
    match_reference_entities(
        "The price of the new smartphone is listed at 999 euros.", [
            "999 euros, currency",
            "999, quantity",
            "euros, currency",
            "smartphone, device",
        ])
    match_reference_entities(
        "The transaction was completed in pesos.", [
            "pesos, currency",
            "transaction, event",
        ])
    match_reference_entities(
        "If you want that, it's two bucks.", [
            "two bucks, currency",
        ])
    match_reference_entities(
        "\"To hell with it!\", he muttered, as he hit submit and plunged his life savings into Bitcoin.", [
            "Bitcoin, currency",
            "hell, concept",
            "life savings, quantity",
            "submit, activity",
        ])


#def test_entity_classifier_case_economic_concept():
#    """
#    Economic concept.
#
#    :return:
#    """
#    match_reference_entities(
#        "I don't get how you can be in business without understanding basic supply and demand.", [
#        ])
#    match_reference_entities(
#        "Inflation can erode the purchasing power of consumers, which has a destabilizing effect on society.", [
#        ])
#    match_reference_entities(
#        "Division of labour has caused a greater increase in production than any other factor.", [
#        ])
#
#
#def test_entity_classifier_case_ethical_existential_philosophical_or_social_concept():
#    """
#    Ethical, existential, moral, philosophical, or social concept
#
#    :return:
#    """
#    match_reference_entities(
#        "The debate over free will versus determinism continues to intrigue modern peoples.", [
#        ])
#    match_reference_entities(
#        "Social justice is a key issue in contemporary political discourse.", [
#        ])
#    match_reference_entities(
#        "The theme of redemption is prevalent throughout the novel.", [
#        ])
#    match_reference_entities(
#        "Were you in Achilles' sandals, what would you choose; glory and a short life or a long life and obscurity?", [
#        ])
#
#
#def test_entity_classifier_case_executive_operational_or_managerial_concept():
#    """
#    Executive, operational, or managerial concept
#
#    :return:
#    """
#    match_reference_entities(
#        "Effective leadership is crucial for the success of any organization.", [
#        ])
#    match_reference_entities(
#        "Operational metrics are important for streamlining inefficient processes.", [
#        ])
#    match_reference_entities(
#        "Corporate leaders want to measure things in the name of efficiency but every measurement incurs a cost.", [
#        ])
#
#
#def test_entity_classifier_case_food_drink_or_other_perishable():
#    """
#    Food, drink, or other perishable consumable.
#
#    :return:
#    """
#    match_reference_entities(
#        "Pass the salmon, lox, and cream cheese.", [
#        ])
#    match_reference_entities(
#        "She enjoyed a refreshing glass of lemonade on a hot summer day.", [
#        ])
#
#
#def test_entity_classifier_case_for_profit_business():
#    """
#    For-Profit Business Organization
#
#    :return:
#    """
#    match_reference_entities(
#        "Apple Inc. announced its latest product launch during the annual conference.", [
#        ])
#
#
#def test_entity_classifier_case_furniture_or_art():
#    """
#    Furniture or art.
#
#    :return:
#    """
#    match_reference_entities(
#        "Pull that chair over so we can use it to prop up this section of the pillow fort.", [
#        ])
#    match_reference_entities(
#        "Go chill on the La-Z-boy.", [
#        ])
#    match_reference_entities(
#        "The art gallery displayed some exquisite Frida Kahlo pieces.", [
#        ])
#    match_reference_entities(
#        "Hide the painting under the stairs.", [
#        ])
#
#
#def test_entity_classifier_case_future_date_or_time():
#    """
#    Future date or time
#
#    :return:
#    """
#    match_reference_entities(
#        "She plans to retire after March 15th next year.", [
#        ])
#    match_reference_entities(
#        "The train comes at 3 PM.", [
#        ])
#
#
#def test_entity_classifier_case_game_or_playful_activity():
#    """
#    Plant or flora.
#
#    :return:
#    """
#    match_reference_entities(
#        "Chess and Go are about strategy and foresight.", [
#        ])
#    match_reference_entities(
#        "Poker has some strategy but is more about reading people and being unreadable to others.", [
#        ])
#    match_reference_entities(
#        "The level of skill in the NBA has become so phenomenal as each generation of athletes push the bar higher.", [
#        ])
#
#
#def test_entity_classifier_case_geographical_location_or_street_address():
#    """
#    Geographical location or street address
#
#    :return:
#    """
#    match_reference_entities(
#        "She lives at 123 Maple Street.", [
#        ])
#    match_reference_entities(
#        "Make sure they see this at 10 Downing.", [
#        ])
#    match_reference_entities(
#        "Hogwarts School of Witchcraft and Wizardry is located in Scotland.", [
#        ])
#    match_reference_entities(
#        "Little did Dorthy know that she would, once again, be stuck in the Land of Oz.", [
#        ])
#
#
#def test_entity_classifier_case_government_organization():
#    """
#    Government organization
#
#    :return:
#    """
#    match_reference_entities(
#        "NASA is responsible for the U.S. space program.", [
#        ])
#    match_reference_entities(
#        "After the hurricane, FEMA was onsite to help with the recovery.", [
#        ])
#
#
#def test_entity_classifier_case_government_program():
#    """
#    Government program
#
#    :return:
#    """
#    match_reference_entities(
#        "Meals on wheels is critical for the seniors who depend on it.", [
#        ])
#    match_reference_entities(
#        "Social Security is an important component of America's social safety net", [
#        ])
#    match_reference_entities(
#        "Many eduction reforms have been controversial, No Child Left Behind and Whole Reading are examples.", [
#        ])
#
#
#def test_entity_classifier_case_humanoid_person_or_personal():
#    """
#    Humanoid person or persona
#
#    :return:
#    """
#    match_reference_entities(
#        "Albert Einstein was a brilliant physicist.", [
#        ])
#    match_reference_entities(
#        "Sherlock Holmes is known for his keen observation skills and his trusted partner, Dr. Watson.", [
#        ])
#
#
#def test_entity_classifier_case_interpersonal_or_relational_concept():
#    """
#    Interpersonal or relational concept
#
#    :return:
#    """
#    match_reference_entities(
#        "Trust is the foundation of a strong relationship and effective communication is how we build it.", [
#        ])
#
#
#def test_entity_classifier_case_job_trade_or_profession():
#    """
#    Job, trade, career, or profession
#
#    :return:
#    """
#    match_reference_entities(
#        "With the shift towards high-tech, traditional trades like carpentry have become comparatively rare.", [
#        ])
#    match_reference_entities(
#        "Although his father got started in the Marines, he wouldn't be allowed follow the same path.", [
#        ])
#
#
#def test_entity_classifier_case_legal_concept():
#    """
#    Material or substance
#
#    :return:
#    """
#    match_reference_entities(
#        ("The rule of law is based on the fundamental principle that \"right is right\", "
#         "rather than \"might is right\"."), [
#        ])
#    match_reference_entities(
#        "A judge is a public servant that provides justice to people harmed by the actions of others.", [
#        ])
#    match_reference_entities(
#        "Judges must remain neutral and unbiased, treating all parties equally without favoritism or prejudice.", [
#        ])
#    match_reference_entities(
#        ("All prospective justices claim to respect stare decisis but whenever an actual chance to tip the scales is "
#         "before them, we have to assume that each would vote based on conscience and principles."), [
#        ])
#
#
#def test_entity_classifier_case_material():
#    """
#    Material or substance
#
#    :return:
#    """
#    match_reference_entities(
#        "The sculpture was crafted from a single block of marble.", [
#        ])
#    match_reference_entities(
#        "The fabric felt soft and luxurious against her skin.", [
#        ])
#    match_reference_entities(
#        ("It is her job to open packages so she did, but she was clearly disturbed when a white powder ejected out of "
#         "this particular container right when she opened it."), [
#        ])
#
#
#def test_entity_classifier_case_musical_instrument():
#    """
#    Musical instrument
#
#    :return:
#    """
#    match_reference_entities(
#        "On the piano, he was like da Vinci with a paint brush.", [
#        ])
#    match_reference_entities(
#        "You should have seen how he made those people dance, playing that hacksaw like Yo-Yo Ma on the cello.", [
#        ])
#
#
#def test_entity_classifier_case_natural_or_artificial_terrain():
#    """
#    Natural or artificial terrain feature
#
#    :return:
#    """
#    match_reference_entities(
#        "They were massacred after the cavalry chased them into a box canyon and rained death from above.", [
#        ])
#    match_reference_entities(
#        "The lake created a natural barrier for the men in the castle, who relied on it's remoteness for protection.", [
#        ])
#
#
#def test_entity_classifier_case_natural_resource():
#    """
#    Natural resource
#
#    :return:
#    """
#    match_reference_entities(
#        "Crude oil is still a vital for many economies, even as the world moves toward renewables.", [
#        ])
#    match_reference_entities(
#        "The lithium deposits in the ancient lake bed could fuel a revolution in battery technology.", [
#        ])
#    match_reference_entities(
#        ("Having used fertilizer containing sewage sludge, Bill was for forced to sell his family's land, "
#         "which is now contaminated and longer useful for agriculture."), [
#        ])
#
#
#def test_entity_classifier_case_non_profit_industry_trace_or_professional_organization():
#    """
#    Non-profit industry, trade or professional organization
#
#    :return:
#    """
#    match_reference_entities(
#        "The American Medical Association sets standards for medical professionals.", [
#        ])
#    match_reference_entities(
#        "The Teamsters are for politicians that support American workers.", [
#        ])
#
#
#def test_entity_classifier_case_non_profit_religious_cultural_or_community_organization():
#    """
#    Non-profit religious, cultural, or community organization
#
#    :return:
#    """
#    match_reference_entities(
#        "The Red Cross provides humanitarian aid worldwide.", [
#        ])
#
#
#def test_entity_classifier_case_past_date_or_time():
#    """
#    Past date or time
#
#    :return:
#    """
#    match_reference_entities(
#        "She says she'll remember that day in the summer of 1969 till her dying breath.", [
#        ])
#
#
#def test_entity_classifier_case_park_or_nature_preserve():
#    """
#    Park or nature preserve.
#
#    :return:
#    """
#    match_reference_entities(
#        ("Yellowstone National Park is home to a variety of wildlife, including some that cause friction between "
#         "the park and nearby farming communities."), [
#        ])
#    match_reference_entities(
#        "Let's meet at the big playground at Lake Elizabeth.", [
#        ])
#    match_reference_entities(
#        ("Take your trailer there, where it's public land that belongs to the Park Service so you can camp "
#         "and sleep in your vehicle."), [
#        ])
#
#
#def test_entity_classifier_case_people_group():
#    """
#    People group
#
#    :return:
#    """
#    match_reference_entities(
#        ("The Norwegian King gave the order to advance, sending first, the shield wall, archers, and "
#         "skirmishers from the Sámi tribes, followed by light spears on Scottish border horses "
#         "and conscripts from nearby towns."), [
#        ])
#    match_reference_entities(
#        ("The band vibed with the audience and with each song, amped up the energy, which you could see "
#         "as the mosh pit circled with ever increasing ferocity."), [
#        ])
#
#
#def test_entity_classifier_case_permanent_building_or_monument():
#    """
#    Permanent building or Monument.
#
#    :return:
#    """
#    match_reference_entities(
#        "The Eiffel Tower is one of the most recognizable landmarks in Paris.", [
#        ])
#
#
#def test_entity_classifier_case_philosophical_concept():
#    """
#    Philosophical concept
#
#    :return:
#    """
#    match_reference_entities(
#        "Existentialism explores the meaning of existence.", [
#        ])
#
#
#def test_entity_classifier_case_plant_or_flora():
#    """
#    Plant or flora.
#
#    :return:
#    """
#    match_reference_entities(
#        "The Amazon rainforest is home to diverse plant species.", [
#        ])
#
#
#def test_entity_classifier_case_political_concept():
#    """
#    Political concept
#
#    :return:
#    """
#    match_reference_entities(
#        "Democracy is based on the principle of equal representation.", [
#        ])
#
#
#def test_entity_classifier_case_psychological_concept():
#    """
#    Psychological concept
#
#    :return:
#    """
#    match_reference_entities(
#        "Cognitive dissonance occurs when beliefs and actions are inconsistent.", [
#        ])
#
#
#def test_entity_classifier_case_quantity_not_related_to_currency():
#    """
#    Quantity not related to currency
#
#    :return:
#    """
#    match_reference_entities(
#        "The recipe calls for two cups of flour.", [
#        ])
#
#
#def test_entity_classifier_case_religious_concept():
#    """
#    Religious concept
#
#    :return:
#    """
#    match_reference_entities(
#        "Karma is a central concept in Hinduism and Buddhism.", [
#        ])
#
#
#def test_entity_classifier_case_ritual_or_tradition():
#    """
#    Ritual or tradition
#
#    :return:
#    """
#    match_reference_entities(
#        "Her heartfelt farewell brought tears to my eyes.", [
#        ])
#
#
#def test_entity_classifier_case_scientific_or_technological_concept():
#    """
#    Scientific or technological concept
#
#    :return:
#    """
#    match_reference_entities(
#        "Quantum mechanics describes the behavior of particles at the atomic level.", [
#        ])
#    match_reference_entities(
#        "Tesla bet big on self-driving without LiDAR.", [
#        ])
#    match_reference_entities(
#        "The new model boasts an impressive battery life.", [
#        ])
#
#
#def test_entity_classifier_case_smell_or_sensation():
#    """
#    Religious concept
#
#    :return:
#    """
#    match_reference_entities(
#        ("The aroma of freshly baked bread filled the whole kitchen, briefly transporting her back to a time and place "
#         "long forgotten, back to her childhood home, to those summer afternoons, baking with her father."), [
#        ])
#    match_reference_entities(
#        ("The chocolate smelled so good and tasted so creamy and sweet that she did feel a little better, even though "
#         "she also felt guilty for finding a little joy when it seemed like everything can only be sad."), [
#        ])
#    match_reference_entities(
#        ("The warm water sent a feeling of relief throughout her body, washing away the aches accumulated over many "
#         "uncomfortable nights, sleeping in the wilderness."), [
#        ])
#    match_reference_entities(
#        "The aged oak of the antique table felt smooth when he brushed his hand against its surface.", [
#        ])
#
#
#def test_entity_classifier_case_social_or_cultural_concept():
#    """
#    Religious concept
#
#    :return:
#    """
#    match_reference_entities(
#        "Cultural diversity and the free flow of people enriches societies.", [
#        ])
#
#
#def test_entity_classifier_case_storage_container():
#    """
#    Storage container.
#
#    :return:
#    """
#    match_reference_entities(
#        "She pulled out a pen and quickly scribbled the new information into her rolodex.", [
#        ])
#    match_reference_entities(
#        "The box was filled with old photographs and letters.", [
#        ])
#    match_reference_entities(
#        "He had a jar for loose change.", [
#        ])
#
#
#def test_entity_classifier_case_temporal_event():
#    """
#    Temporal event.
#
#    :return:
#    """
#    match_reference_entities(
#        "The solar eclipse will occur next month.", [
#        ])
#    match_reference_entities(
#        "The grand opening of the new museum is in a year.", [
#        ])
#    match_reference_entities(
#        "The signing of the Declaration of Independence was a pivotal moment in history.", [
#        ])
#    match_reference_entities(
#        "The Olympics are held every four years.", [
#        ])
#
#
#def test_entity_classifier_case_temporary_structure():
#    """
#    Storage container.
#
#    :return:
#    """
#    match_reference_entities(
#        "The circus tent will set up in the outer field for the next three weeks.", [
#        ])
#    match_reference_entities(
#        "The construction crew erected scaffolding around the building.", [
#        ])
#
#
#def test_entity_classifier_case_utensil_instrument_or_machinery():
#    """
#    Utensil, instrument, machinery, or other mechanical tool.
#
#    :return:
#    """
#    match_reference_entities(
#        "The can opener is in the top drawer.", [
#        ])
#    match_reference_entities(
#        "The microscope is essential for studying microorganisms.", [
#        ])
#    match_reference_entities(
#        "Ain't nobody plays the trumpet like Miles Davis.", [
#        ])
#
#
#def test_entity_classifier_case_vehicle():
#    """
#    Vehicle
#
#    :return:
#    """
#    match_reference_entities(
#        "Move your car, the garbage truck is coming.", [
#        ])
#    match_reference_entities(
#        "Tesla's Model Y is now their top seller.", [
#        ])


if __name__ == "__main__":
    from observability.logging import setup_logging
    setup_logging()

    #test_entity_classifier_case_abstract_ability_or_attribute()
    #test_entity_classifier_case_animal_or_non_humanoid()
    #test_entity_classifier_case_article_book_document_or_other_text()
    #test_entity_classifier_case_artistic_or_literary_concept()
    #test_entity_classifier_case_audio_image_video_or_other_media()
    #test_entity_classifier_case_clothing_shoes_or_jewelry()
    #test_entity_classifier_case_comment_message_letter_or_communication_artifact()
    #test_entity_classifier_case_computer_phone_or_electronic_device()
    #test_entity_classifier_case_construction_or_industrial_input()
    #test_entity_classifier_case_city_county_and_other_localities()
    #test_entity_classifier_case_crime_terror_or_paramilitary()
    test_entity_classifier_case_currency()
