import logging

from bot.lang.classifiers import entity_classifier, entity_role_classifier

logger = logging.getLogger(__name__)


def match_reference_entities(test_sentence, reference_classification):
    new_classification = entity_classifier.classify(test_sentence)
    logger.info(f"{test_sentence} {new_classification}")
    assert "entities" in new_classification
    for entity in new_classification["entities"]:
        assert "entity" in entity
        assert "entity_type" in entity
        signature_parts = [
            entity["entity"],
            entity["entity_type"],
        ]
        check_signature = ", ".join(signature_parts)
        assert check_signature in reference_classification, check_signature


def test_entity_classifier_case_abstract_ability_or_attribute():
    """
    Ability, attribute, feature, or trait.

    :return:
    """
    match_reference_entities(
        "Her kindness is truly admirable and inspiring.", [
            "Her kindness, ethical, existential, moral, philosophical, or social concept",
            "inspiring, abstract ability or attribute",
            "kindness, abstract ability or attribute",
            "kindness, ethical, existential, moral, philosophical, or social concept",
            "kindness, psychological concept",
        ])
    match_reference_entities(
        "Intelligence is often measured by one's ability to solve complex problems.", [
            "Intelligence, abstract ability or attribute",
            "complex problems, ambiguous or highly abstract concept",
        ])


def test_entity_classifier_case_animal_or_non_humanoid():
    """
    Animal or Non-Humanoid Creature.

    :return:
    """
    match_reference_entities(
        "The elephant wandered through the savannah, searching for water.", [
            "elephant, animal or non-humanoid creature",
            "savannah, geographical concept",
            "savannah, terrain feature, natural or artificial",
            "water, food, drink, or other perishable consumable",
            "water, material or substance",
            "water, natural resource",
        ])
    match_reference_entities(
        "In the dense jungle, a jaguar prowled silently through the underbrush.", [
            "dense jungle, terrain feature, natural or artificial",
            "jaguar, animal or non-humanoid creature",
            "jungle, terrain feature, natural or artificial",
            "underbrush, terrain feature, natural or artificial",
        ])
    match_reference_entities(
        "Dolphins are known for their intelligence and playful behavior.", [
            "Dolphins, animal or non-humanoid creature",
            "intelligence, abstract ability or attribute",
            "playful behavior, abstract ability or attribute",
        ])


def test_entity_classifier_case_article_book_document_or_other_text():
    """
    Article, book, document, post or other textual artifact.

    :return:
    """
    match_reference_entities(
        "I read an interesting book on artificial intelligence last night.", [
            "artificial intelligence, scientific or technological concept",
            "book, article, book, document, post, or other text-based artifact",
            "photographs, ",
        ])
    match_reference_entities(
        "The research paper provided new insights into climate change.", [
            "climate change, ambiguous or highly abstract concept",
            "climate change, ethical, existential, moral, philosophical, or social concept",
            "climate change, scientific or technological concept",
            "research paper, article, book, document, post, or other text-based artifact",
        ])
    match_reference_entities(
        "I just finished reading the assigned chapter from \"To Kill a Mockingbird\".", [
            "To Kill a Mockingbird, article, book, document, post, or other text-based artifact",
        ])


def test_entity_classifier_case_artistic_or_literary_concept():
    """
    Artistic or literary concept

    :return:
    """
    match_reference_entities(
        "Surrealism challenges the boundaries of reality and imagination.", [
            "Surrealism, artistic or literary concept",
            "reality, ambiguous or highly abstract concept",
            "imagination, ambiguous or highly abstract concept",
        ])
    match_reference_entities(
        "The works in this period of the painter's life is filled with remorse and repentance related imagery.", [
            "painter, humanoid person or individual persona",
            "remorse and repentance related imagery, artistic or literary concept",
            "remorse and repentance, ethical, existential, moral, philosophical, or social concept",
            "remorse, ethical, existential, moral, philosophical, or social concept",
            "repentance, ethical, existential, moral, philosophical, or social concept",
        ])


def test_entity_classifier_case_audio_image_video_or_other_media():
    """
    Audio, image, video, or other captured or recorded media artifact

    :return:
    """
    match_reference_entities(
        ("She used an audio recorder to tape the interview with the cartel member, "
         "after being taken to an unspecified location."), [
            "audio recorder, computer, phone, or personal electronic device",
            "cartel member, humanoid person or individual persona",
            "interview, comment, message, letter or other communication artifact",
            "interview, event",
            "unspecified location, unspecified spatial location or spatial concept",
        ])
    match_reference_entities(
        "The podcast episode explored the history of jazz music.", [
            "jazz music, artistic or literary concept",
            "podcast episode, article, book, document, post, or other text-based artifact",
            "podcast episode, audio, image, video, or other recorded media artifact"
        ])
    match_reference_entities(
        "A viral video of a cat playing the piano captivated millions online.", [
            "cat, animal or non-humanoid creature",
            "piano, musical instrument",
            "viral video, audio, image, video, or other recorded media artifact",
        ])


def test_entity_classifier_case_clothing_shoes_or_jewelry():
    """
    Clothing, Shoes, or Jewelry.

    :return:
    """
    match_reference_entities(
        "She wore a stunning diamond necklace to the gala.", [
            "diamond necklace, clothing, shoes, or jewelry",
            "gala, event",
        ])
    match_reference_entities(
        "His wore designer Air Jordans to the game.", [
            "Air Jordans, clothing, shoes, or jewelry",
        ])


def test_entity_classifier_case_comment_message_letter_or_communication_artifact():
    """
    Comment, message, letter, or other communication artifact

    :return:
    """
    match_reference_entities(
        "Right before she disappeared, she sent me three messages", [
            "messages, comment, message, letter or other communication artifact",
            "she, humanoid person or individual persona",
            "three messages, comment, message, letter or other communication artifact",
        ])
    match_reference_entities(
        "The email contained important updates about the project.", [
            "email, comment, message, letter or other communication artifact",
            "project, initiative or objective",
        ])
    match_reference_entities(
        "Right as the riot police began controlling the crowd, she shot off a tweet to her followers.", [
            "crowd, interpersonal or relational concept",
            "crowd, people group",
            "followers, interpersonal or relational concept",
            "followers, people group",
            "riot police, organization",
            "tweet, comment, message, letter or other communication artifact",
        ])


def test_entity_classifier_case_computer_phone_or_electronic_device():
    """
    Computer, phone, or personal electronic device.

    :return:
    """
    match_reference_entities(
        "He upgraded to the latest smartphone model.", [
            "latest smartphone model, computer, phone, or personal electronic device",
            "smartphone, computer, phone, or personal electronic device",
        ])
    match_reference_entities(
        "His laptop fell down three flights of stairs and crashed against a concrete surface.", [
            "concrete surface, material or substance",
            "concrete surface, terrain feature, natural or artificial",
            "laptop, computer, phone, or personal electronic device",
            "stairs, terrain feature, natural or artificial",
            "three flights of stairs, terrain feature, natural or artificial",
        ])
    match_reference_entities(
        "He had been very excited about his Google glasses but he stopped wearing them after the backlash.", [
            "Google glasses, computer, phone, or personal electronic device",
        ])


def test_entity_classifier_case_construction_or_industrial_input():
    """
    Construction or industrial input.

    :return:
    """
    match_reference_entities(
        "The building was constructed using high-quality steel beams.", [
            "building, permanent building or monument",
            "steel beams, material or substance",
        ])
    match_reference_entities(
        "Cement is as important today as it was to the Romans.", [
            "Cement, material or substance",
            "Cement, natural resource",
            "Romans, people group",
        ])
    match_reference_entities(
        "The renewed interest in nuclear power due to AI has sparked a rally in palladium prices.", [
            "AI, scientific or technological concept",
            "nuclear power, scientific or technological concept",
            "palladium prices, economic concept",
        ])


def test_entity_classifier_case_city_county_and_other_localities():
    """
    Cities and other localities.

    :return:
    """
    match_reference_entities(
        "He used to live in Alameda County but moved further south along the 880.", [
            "Alameda County, county",
            "880, geographical concept",
        ])


def test_entity_classifier_case_crime_terror_or_paramilitary():
    """
    Crime, terror, or paramilitary organization.

    :return:
    """
    match_reference_entities(
        "The investigation revealed links to the notorious street gang.", [
            "street gang, organization",
            "street gang, people group",
        ])
    match_reference_entities(
        "Having pushed Asad out of Syria, Hayat Tahrir al-Sham now must form a new legitimate government.", [
            "Asad, humanoid person or individual persona",
            "Hayat Tahrir al-Sham, organization",
            "Syria, country",
        ])


def test_entity_classifier_case_currency():
    """
    Currency.

    :return:
    """
    match_reference_entities(
        "The price of the new smartphone is listed at 999 euros.", [
            "999 euros, currency",
            "smartphone, computer, phone, or personal electronic device",
        ])
    match_reference_entities(
        "The transaction was completed in pesos.", [
            "pesos, currency",
            "transaction, economic concept",
        ])
    match_reference_entities(
        "If you want that, it's two bucks", [
            "two bucks, currency",
        ])


def test_entity_classifier_case_economic_concept():
    """
    Economic concept.

    :return:
    """
    match_reference_entities(
        "I don't get how you can be in business without understanding basic supply and demand.", [
            "business, organization",
            "supply and demand, economic concept",
        ])
    match_reference_entities(
        "Inflation can erode the purchasing power of consumers, which as a destabilizing effect on society.", [
            "Inflation, economic concept",
            "consumers, people group",
            "purchasing power, economic concept",
            "destabilizing effect on society, ethical, existential, moral, philosophical, or social concept",
        ])
    match_reference_entities(
        "Division of labour has caused a greater increase in production than any other factor.", [
            "Division of labour, economic concept",
            "production, economic concept",
        ])


def test_entity_classifier_case_ethical_existential_philosophical_or_social_concept():
    """
    Ethical, existential, moral, philosophical, or social concept

    :return:
    """
    match_reference_entities(
        "The debate over free will versus determinism continues to intrigue modern peoples.", [
            "determinism, ethical, existential, moral, philosophical, or social concept",
            "free will, ethical, existential, moral, philosophical, or social concept",
            "modern peoples, people group",
        ])
    match_reference_entities(
        "Social justice is a key issue in contemporary political discourse.", [
            "Social justice, ethical, existential, moral, philosophical, or social concept",
            "Social justice, political concept",
            "contemporary political discourse, ambiguous or highly abstract concept",
            "contemporary political discourse, ethical, existential, moral, philosophical, or social concept",
            "contemporary political discourse, political concept",
            "political discourse, political concept",
        ])
    match_reference_entities(
        "The theme of redemption is prevalent throughout the novel.", [
            "novel, article, book, document, post, or other text-based artifact",
            "redemption, ethical, existential, moral, philosophical, or social concept",
        ])


def test_entity_classifier_case_executive_operational_or_managerial_concept():
    """
    Executive, operational, or managerial concept

    :return:
    """
    match_reference_entities(
        "Effective leadership is crucial for the success of any organization.", [
            "leadership, ethical, existential, moral, philosophical, or social concept",
            "leadership, executive, operational, or managerial concept",
            "organization, organization",
        ])
    match_reference_entities(
        "Operational metrics are important for streamlining inefficient processes.", [
            "inefficient processes, ethical, existential, moral, philosophical, or social concept",
            "inefficient processes, executive, operational, or managerial concept",
            "operational metrics, executive, operational, or managerial concept",
        ])
    match_reference_entities(
        "Corporate leaders want to measure things in the name of efficiency but every measurement incurs a cost.", [
            "Corporate leaders, humanoid person or individual persona",
            "cost, economic concept",
            "efficiency, ambiguous or highly abstract concept",
            "efficiency, economic concept",
            "efficiency, ethical, existential, moral, philosophical, or social concept",
            "measurement, abstract ability or attribute",
            "measurement, scientific or technological concept",
        ])


def test_entity_classifier_case_food_drink_or_other_perishable():
    """
    Food, drink, or other perishable consumable.

    :return:
    """
    match_reference_entities(
        "Pass the salmon, lox, and cream cheese.", [
            "salmon, food, drink, or other perishable consumable",
            "lox, food, drink, or other perishable consumable",
            "cream cheese, food, drink, or other perishable consumable",
        ])
    match_reference_entities(
        "She enjoyed a refreshing glass of lemonade on a hot summer day.", [
            "lemonade, food, drink, or other perishable consumable",
            "summer day, event",
            "summer day, future date or time",
            "summer day, time duration",
        ])


def test_entity_classifier_case_for_profit_business():
    """
    For-Profit Business Organization

    :return:
    """
    match_reference_entities(
        "Apple Inc. announced its latest product launch during the annual conference.", [
            "Apple Inc., organization",
            "annual conference, event",
            "latest product launch, article, book, document, post, or other text-based artifact",
            "latest product launch, initiative or objective",
            "latest product launch, event",
        ])


def test_entity_classifier_case_furniture_or_art():
    """
    Furniture or art.

    :return:
    """
    match_reference_entities(
        "Pull that chair over so we can use it to prop up this section of the pillow fort.", [
            "chair, furniture or art",
            "pillow fort, game or playful activity",
            "pillow fort, temporary structure",
        ])
    match_reference_entities(
        "Go chill on the La-Z-boy.", [
            "La-Z-boy, furniture or art"
        ])
    match_reference_entities(
        "The art gallery displayed some exquisite Frida Kahlo pieces.", [
            "Frida Kahlo, humanoid person or individual persona",
            "art gallery, permanent building or monument",
            "pieces, artistic or literary concept",
        ])
    match_reference_entities(
        "Hide the painting under the stairs.", [
            "painting, furniture or art",
            "stairs, permanent building or monument",
            "stairs, terrain feature, natural or artificial",
        ])


def test_entity_classifier_case_future_date_or_time():
    """
    Future date or time

    :return:
    """
    match_reference_entities(
        "She plans to retire after March 15th next year.", [
            "March 15th next year, future date or time",
            "March 15th, future date or time",
            "next year, future date or time",
        ])
    match_reference_entities(
        "The train comes at 3 PM.", [
            "3 PM, future date or time",
            "train, vehicle",
        ])


def test_entity_classifier_case_geographical_location_or_street_address():
    """
    Geographical location or street address

    :return:
    """
    match_reference_entities(
        "She lives at 123 Maple Street.", [
            "123 Maple Street, postal code or street address",
        ])
    match_reference_entities(
        "Make sure they see this at 10 Downing.", [
            "10 Downing, postal code or street address",
            "10 Downing, permanent building or monument",
        ])
    match_reference_entities(
        "Hogwarts School of Witchcraft and Wizardry is located in Scotland.", [
            "Hogwarts School of Witchcraft and Wizardry, permanent building or monument",
            "Hogwarts School of Witchcraft and Wizardry, organization",
            "Scotland, country",
        ])
    match_reference_entities(
        "Little did Dorthy know that she would, once again, be stuck in the Land of Oz.", [
            "Dorthy, humanoid person or individual persona",
            "Land of Oz, geographical concept",
        ])


def test_entity_classifier_case_government_organization():
    """
    Government organization

    :return:
    """
    match_reference_entities(
        "NASA is responsible for the U.S. space program.", [
            "NASA, organization",
            "U.S. space program, government program",
            "U.S. space program, initiative or objective",
        ])
    match_reference_entities(
        "After the hurricane, FEMA was onsite to help with the recovery.", [
            "FEMA, organization",
            "hurricane, natural phenomenon",
            "recovery, initiative or objective",
        ])


def test_entity_classifier_case_government_program():
    """
    Government organization

    :return:
    """
    match_reference_entities(
        "Meals on wheels is critical for the seniors who depend on it.", [
            "Meals on Wheels, organization",
            "seniors, people group",
        ])
    match_reference_entities(
        "Social Security is an important component of America's social safety net", [
            "America, country",
            "Social Security, government program",
            "social safety net, ethical, existential, moral, philosophical, or social concept",
        ])
    match_reference_entities(
        "Many eduction reforms have been controversial, No Child Left Behind and Whole Reading are examples.", [
            "No Child Left Behind, educational concept",
            "No Child Left Behind, government program",
            "Whole Reading, educational concept",
        ])


def test_entity_classifier_case_humanoid_person_or_personal():
    """
    Humanoid person or persona

    :return:
    """
    match_reference_entities(
        "Albert Einstein was a brilliant physicist.", [
            "Albert Einstein, humanoid person or individual persona",
            "physicist, job, trade, career, or profession",
        ])
    match_reference_entities(
        "Sherlock Holmes is known for his keen observation skills and his trusted partner, Dr. Watson.", [
            "Sherlock Holmes, humanoid person or individual persona",
            "Dr. Watson, humanoid person or individual persona",
            "keen observation skills, abstract ability or attribute",
            "observation skills, abstract ability or attribute",
        ])


def test_entity_classifier_case_interpersonal_or_relational_concept():
    """
    Interpersonal or relational concept

    :return:
    """
    match_reference_entities(
        "Trust is the foundation of a strong relationship and effective communication is how we build it.", [
            "Trust, ethical, existential, moral, philosophical, or social concept",
            "communication, ethical, existential, moral, philosophical, or social concept",
            "communication, interpersonal or relational concept",
            "relationship, interpersonal or relational concept",
        ])


def test_entity_classifier_case_job_trade_or_profession():
    """
    Job, trade, career, or profession

    :return:
    """
    match_reference_entities(
        "With the shift towards high-tech, traditional trades like carpentry have become comparatively rare.", [
            "carpentry, job, trade, career, or profession",
            "high-tech, scientific or technological concept",
        ])
    match_reference_entities(
        "Although his father got started in the Marines, he wouldn't be allowed follow the same path.", [
            "Marines, military concept",
            "father, humanoid person or individual persona",
        ])


def test_entity_classifier_case_musical_instrument():
    """
    Musical instrument

    :return:
    """
    match_reference_entities(
        "On the piano, he was like da Vinci with a paint brush.", [
            "da Vinci, humanoid person or individual persona",
            "paint brush, utensil, instrument, machinery, or other mechanical tool",
            "piano, musical instrument",
        ])
    match_reference_entities(
        "You should have seen how he made those people dance, playing that hacksaw like Yo-Yo Ma on the cello.", [
            "Yo-Yo Ma, humanoid person or individual persona",
            "cello, musical instrument",
            "hacksaw, musical instrument",
            "hacksaw, utensil, instrument, machinery, or other mechanical tool",
        ])


def test_entity_classifier_case_natural_or_artificial_terrain():
    """
    Natural or artificial terrain feature

    :return:
    """
    match_reference_entities(
        "They were massacred after the cavalry chased them into a box canyon and rained death from above.", [
            "box canyon, terrain feature, natural or artificial",
            "cavalry, military concept",
            "death, abstract ability or attribute",
            "death, ambiguous or highly abstract concept",
        ])
    match_reference_entities(
        "The lake created a natural barrier for the men in the castle, who relied on it's remoteness for protection.", [
            "castle, permanent building or monument",
            "lake, terrain feature, natural or artificial",
            "men, humanoid person or individual persona",
            "men, people group",
        ])


def test_entity_classifier_case_natural_resource():
    """
    Natural resource

    :return:
    """
    match_reference_entities(
        "Crude oil is still a vital for many economies, even as the world moves toward renewables.", [
            "Crude oil, natural resource",
            "economies, economic concept",
            "renewables, economic concept",
            "renewables, ethical, existential, moral, philosophical, or social concept",
            "renewables, natural phenomenon",
            "renewables, natural resource",
            "renewables, scientific or technological concept",
        ])
    match_reference_entities(
        "The lithium deposits in the ancient lake bed could fuel a revolution in battery technology.", [
            "ancient lake bed, geographical concept",
            "ancient lake bed, terrain feature, natural or artificial",
            "battery technology, scientific or technological concept",
            "lithium deposits, natural resource",
        ])
    match_reference_entities(
        ("Having used fertilizer containing sewage sludge, Bill was for forced to sell his family's land, "
         "which is now contaminated and longer useful for agriculture."), [
            "Bill, humanoid person or individual persona",
            "agriculture, economic concept",
            "agriculture, job, trade, career, or profession",
            "contaminated land, geographical concept",
            "contaminated land, terrain feature, natural or artificial",
            "family's land, geographical concept",
            "fertilizer containing sewage sludge, material or substance",
            "fertilizer containing sewage sludge, food, drink, or other perishable consumable",
        ])


def test_entity_classifier_case_non_profit_industry_trace_or_professional_organization():
    """
    Non-profit industry, trade or professional organization

    :return:
    """
    match_reference_entities(
        "The American Medical Association sets standards for medical professionals.", [
            "American Medical Association, organization",
            "medical professionals, humanoid person or individual persona",
            "medical professionals, job, trade, career, or profession",
            "medical professionals, people group",
        ])
    match_reference_entities(
        "The Teamsters are for politicians that support American workers.", [
            "American workers, people group",
            "Teamsters, organization",
            "The Teamsters, organization",
            "The Teamsters, political concept",
            "politicians, humanoid person or individual persona",
        ])


def test_entity_classifier_case_non_profit_religious_cultural_or_community_organization():
    """
    Non-profit religious, cultural, or community organization

    :return:
    """
    match_reference_entities(
        "The Red Cross provides humanitarian aid worldwide.", [
            "Red Cross, organization",
            "humanitarian aid, economic concept",
            "humanitarian aid, ethical, existential, moral, philosophical, or social concept",
            "humanitarian aid, initiative or objective",
            "worldwide, geographical concept",
            "worldwide, unspecified spatial location or spatial concept",
        ])


def test_entity_classifier_case_past_date_or_time():
    """
    Past date or time

    :return:
    """
    match_reference_entities(
        "She says she'll remember that day in the summer of 1969 till her dying breath.", [
            "summer of 1969, past date or time",
        ])


def test_entity_classifier_case_people_group():
    """
    People group

    :return:
    """
    match_reference_entities(
        ("The Norwegian King gave the order to advance, sending first, the shield wall, archers, and "
         "skirmishers from the Sámi tribes, followed by light spears on Scottish border horses "
         "and conscripts from nearby towns."), [
            "Norwegian King, humanoid person or individual persona",
            "Scottish border horses, animal or non-humanoid creature",
            "Sámi tribes, people group",
            "light spears, utensil, instrument, machinery, or other mechanical tool",
        ])
    match_reference_entities(
        ("The band vibed with the audience and with each song, amped up the energy, which you could see "
         "as the mosh pit circled with ever increasing ferocity."), [
            "audience, people group",
            "band, organization",
            "mosh pit, event",
            "mosh pit, game or playful activity",
            "mosh pit, interpersonal or relational concept",
            "song, artistic or literary concept",

            # TODO: probably due to interpretation as a hole in the ground
            "mosh pit, temporary structure",
            "mosh pit, terrain feature, natural or artificial",
        ])


def test_entity_classifier_case_permanent_building_or_monument():
    """
    Permanent building or Monument.

    :return:
    """
    match_reference_entities(
        "The Eiffel Tower is one of the most recognizable landmarks in Paris.", [
            "Eiffel Tower, permanent building or monument",
            "Paris, city, county, district, or other municipality",
        ])


def test_entity_classifier_case_philosophical_concept():
    """
    Philosophical concept

    :return:
    """
    match_reference_entities(
        "Existentialism explores the meaning of existence.", [
            "Existentialism, ethical, existential, moral, philosophical, or social concept",
            "meaning of existence, ambiguous or highly abstract concept",
            "meaning of existence, ethical, existential, moral, philosophical, or social concept",
        ])


def test_entity_classifier_case_plant_or_flora():
    """
    Plant or flora.

    :return:
    """
    match_reference_entities(
        "The Amazon rainforest is home to diverse plant species.", [
            "Amazon rainforest, geographical concept",
            "Amazon rainforest, natural resource",
            "plant species, plant or flora",
        ])


def test_entity_classifier_case_political_concept():
    """
    Political concept

    :return:
    """
    match_reference_entities(
        "Democracy is based on the principle of equal representation.", [
            "Democracy, political concept",
            "equal representation, ethical, existential, moral, philosophical, or social concept",
        ])


def test_entity_classifier_case_psychological_concept():
    """
    Psychological concept

    :return:
    """
    match_reference_entities(
        "Cognitive dissonance occurs when beliefs and actions are inconsistent.", [
            "Cognitive dissonance, ethical, existential, moral, philosophical, or social concept",
            "Cognitive dissonance, psychological concept",
            "actions, ambiguous or highly abstract concept",
            "actions, ethical, existential, moral, philosophical, or social concept",
            "beliefs, ambiguous or highly abstract concept",
            "beliefs, ethical, existential, moral, philosophical, or social concept",
        ])


def test_entity_classifier_case_quantity_not_related_to_currency():
    """
    Quantity not related to currency

    :return:
    """
    match_reference_entities(
        "The recipe calls for two cups of flour.", [
            "flour, food, drink, or other perishable consumable",
            "flour, material or substance",
            "two cups, quantity not related to currency or time",
        ])


def test_entity_classifier_case_religious_concept():
    """
    Religious concept

    :return:
    """
    match_reference_entities(
        "Karma is a central concept in Hinduism and Buddhism.", [
            "Buddhism, religious concept",
            "Hinduism, religious concept",
            "Karma, ethical, existential, moral, philosophical, or social concept",
        ])


def test_entity_classifier_case_scientific_or_technological_concept():
    """
    Scientific or technological concept

    :return:
    """
    match_reference_entities(
        "Quantum mechanics describes the behavior of particles at the atomic level.", [
            "Quantum mechanics, scientific or technological concept",
            "atomic level, abstract ability or attribute",
            "atomic level, natural phenomenon",
            "atomic level, scientific or technological concept",
            "particles, abstract ability or attribute",
            "particles, natural phenomenon",
            "particles, subatomic to atomic scale phenomenon",

            # TODO: probably confusion due to "level", potentially referring to a place
            "atomic level, geographical concept",
        ])
    match_reference_entities(
        "Tesla bet big on self-driving without LiDAR.", [
            "LiDAR, scientific or technological concept",
            "Tesla, organization",
            "self-driving, scientific or technological concept",
        ])


def test_entity_classifier_case_social_or_cultural_concept():
    """
    Religious concept

    :return:
    """
    match_reference_entities(
        "Cultural diversity and the free flow of people enriches societies.", [
            "Cultural diversity, ethical, existential, moral, philosophical, or social concept",
            "free flow of people, ethical, existential, moral, philosophical, or social concept",
            "societies, ambiguous or highly abstract concept",
            "societies, people group",
        ])


def test_entity_classifier_case_storage_container():
    """
    Storage container.

    :return:
    """
    match_reference_entities(
        "She pulled out a pen and quickly scribbled the new information into her rolodex.", [
            "pen, utensil, instrument, machinery, or other mechanical tool",
            "rolodex, storage container or organizational tool",
        ])
    match_reference_entities(
        "The box was filled with old photographs and letters.", [
            "box, storage container or organizational tool",
            "letters, comment, message, letter or other communication artifact",
            "photographs, audio, image, video, or other recorded media artifact",
        ])
    match_reference_entities(
        "He had a jar for loose change.", [
            "jar, storage container or organizational tool",
            "loose change, currency",
        ])


def test_entity_classifier_case_temporal_event():
    """
    Temporal event.

    :return:
    """
    match_reference_entities(
        "The solar eclipse will occur next month.", [
            "solar eclipse, event",
            "next month, future date or time",
        ])
    match_reference_entities(
        "The grand opening of the new museum is in a year.", [
            "a year, future date or time",
            "a year, time duration",
            "grand opening, event",
            "museum, permanent building or monument",
            "new museum, permanent building or monument",
        ])
    match_reference_entities(
        "The signing of the Declaration of Independence was a pivotal moment in history.", [
            "Declaration of Independence, article, book, document, post, or other text-based artifact",
            "history, ambiguous or highly abstract concept",
            "signing, event",
        ])
    match_reference_entities(
        "The Olympics are held every four years.", [
            "Olympics, event",
            "four years, time duration",
        ])


def test_entity_classifier_case_temporary_structure():
    """
    Storage container.

    :return:
    """
    match_reference_entities(
        "The circus tent will set up in the outer field for the next three weeks.", [
            "circus tent, temporary structure",
            "outer field, unspecified spatial location or spatial concept",
            "three weeks, time duration",
        ])
    match_reference_entities(
        "The construction crew erected scaffolding around the building.", [
            "building, permanent building or monument",
            "construction crew, organization",
            "scaffolding, temporary structure",
        ])


def test_entity_classifier_case_utensil_instrument_or_machinery():
    """
    Utensil, instrument, machinery, or other mechanical tool.

    :return:
    """
    match_reference_entities(
        "The can opener is in the top drawer.", [
            "can opener, utensil, instrument, machinery, or other mechanical tool",
            "top drawer, furniture or art",
            "top drawer, storage container or organizational tool",
        ])
    match_reference_entities(
        "The microscope is essential for studying microorganisms.", [
            "microorganisms, animal or non-humanoid creature",
            "microscope, utensil, instrument, machinery, or other mechanical tool",
        ])
    match_reference_entities(
        "Ain't nobody plays the trumpet like Miles Davis.", [
            "Miles Davis, humanoid person or individual persona",
            "trumpet, musical instrument",
        ])


def test_entity_classifier_case_vehicle():
    """
    Vehicle

    :return:
    """
    match_reference_entities(
        "Move your car, the garbage truck is coming.", [
            "car, vehicle",
            "garbage truck, vehicle",
        ])
    match_reference_entities(
        "Tesla's Model Y is now their top seller.", [
            "Model Y, vehicle",
            "Tesla's Model Y, vehicle",
            "Tesla, organization",
        ])


if __name__ == "__main__":
    from observability.logging import setup_logging
    setup_logging()
