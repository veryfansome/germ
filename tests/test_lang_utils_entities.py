import logging

from bot.lang.classifiers import get_entity_type_classifier

logger = logging.getLogger(__name__)


def match_reference_entities(test_sentence, reference_classification):
    new_classification = get_entity_type_classifier().classify(test_sentence)
    logger.info(f"{test_sentence} {new_classification}")
    assert "entities" in new_classification
    for entity in new_classification["entities"]:
        assert "entity" in entity
        assert "entity_type" in entity
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
            "savannah, area",
            "savannah, terrain",
            "water, material or substance",
        ])
    match_reference_entities(
        "In the dense jungle, a jaguar prowled silently through the underbrush.", [
            "jaguar, fauna",
            "jungle, area",
            "jungle, terrain",
            "underbrush, flora",
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
            "period, duration",
            "period, time",
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
            "location, place",
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


def test_entity_classifier_case_city_county_and_other_localities():
    """
    Cities and other localities.

    :return:
    """
    match_reference_entities(
        "He used to live in Alameda County but moved further south along the 880.", [
            "880, area",
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
            "Manchuria, place",
            "South Manchurian Railway, place",
            "South Manchurian Railway, structure",
            "invasion, activity",
            "invasion, event",
            "invasion of Manchuria, event",
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
            "three messages, communication",
            "three messages, concept",
            "three messages, event",
            "three messages, media",
            "three messages, quantity",
        ])
    match_reference_entities(
        "The email contained important updates about the project.", [
            "email, communication",
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
    Construction or industrial input

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


def test_entity_classifier_case_contains_code():
    """
    Contains code

    :return:
    """
    match_reference_entities(
        "I tried your suggestion but I got and error, `RuntimeError: Failed to open database`.", [
            "RuntimeError, concept",
            "database, container",
            "error, concept",
            "suggestion, concept",
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
            "life savings, currency",
            "life savings, quantity",
            "submit, activity",
        ])


def test_entity_classifier_case_economic_concept():
    """
    Economic concept.

    :return:
    """
    match_reference_entities(
        "I don't get how you can be in business without understanding basic supply and demand.", [
            "Inflation, concept",
            "business, concept",
            "supply and demand, concept",
        ])
    match_reference_entities(
        "Inflation can erode the purchasing power of consumers, which has a destabilizing effect on society.", [
            "Inflation, concept",
            "consumers, person",
            "destabilizing effect, concept",
            "destabilizing effect, phenomenon",
            "purchasing power, concept",
            "society, concept",
        ])
    match_reference_entities(
        "Division of labour has caused a greater increase in production than any other factor.", [
            "Division of labour, concept",
            "division of labour, concept",
            "factor, concept",
            "production, concept",
        ])


def test_entity_classifier_case_ethical_existential_philosophical_or_social_concept():
    """
    Ethical, existential, moral, philosophical, or social concept

    :return:
    """
    match_reference_entities(
        "The debate over free will versus determinism continues to intrigue modern peoples.", [
            "determinism, concept",
            "free will, concept",
            "modern peoples, person",
        ])
    match_reference_entities(
        "Social justice is a key issue in contemporary political discourse.", [
            "Social justice, concept",
            "contemporary political discourse, concept",
            "political discourse, concept",
        ])
    match_reference_entities(
        "The theme of redemption is prevalent throughout the novel.", [
            "novel, media",
            "redemption, concept",
        ])
    match_reference_entities(
        "Were you in Achilles' sandals, what would you choose; glory and a short life or a long life and obscurity?", [
            "Achilles, person",
            "Achilles' sandals, concept",
            "sandals, apparel",
            "glory, concept",
            "long life, concept",
            "long life, duration",
            "obscurity, concept",
            "short life, concept",
            "short life, duration",
        ])
    match_reference_entities(
        "Existentialism explores the meaning of existence.", [
            "Existentialism, concept",
            "meaning of existence, concept",
        ])


def test_entity_classifier_case_executive_operational_or_managerial_concept():
    """
    Executive, operational, or managerial concept

    :return:
    """
    match_reference_entities(
        "Effective leadership is crucial for the success of any organization.", [
            "leadership, concept",
            "organization, concept",
            "organization, organization",
        ])
    match_reference_entities(
        "Operational metrics are important for streamlining inefficient processes.", [
            "Operational metrics, concept",
            "inefficient processes, concept",
            "operational metrics, concept",
        ])
    match_reference_entities(
        "Corporate leaders want to measure things in the name of efficiency but every measurement incurs a cost.", [
            "Corporate leaders, organization",
            "Corporate leaders, person",
            "cost, concept",
            "cost, currency",
            "efficiency, concept",
            "measurement, activity",
            "measurement, concept",
        ])


def test_entity_classifier_case_food_drink_or_other_perishable():
    """
    Food, drink, or other perishable consumable.

    :return:
    """
    match_reference_entities(
        "Pass the salmon, lox, and cream cheese.", [
            "cream cheese, food",
            "lox, food",
            "salmon, fauna",
            "salmon, food",
        ])
    match_reference_entities(
        "She enjoyed a refreshing glass of lemonade on a hot summer day.", [
            "day, date or time",
            "lemonade, food",
            "summer, season",
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
            "latest product launch, event",
            "product launch, event",
        ])


def test_entity_classifier_case_furniture_or_art():
    """
    Furniture or art.

    :return:
    """
    match_reference_entities(
        "Pull that chair over so we can use it to prop up this section of the pillow fort.", [
            "chair, furniture",
            "chair, container",
            "pillow fort, structure",
        ])
    match_reference_entities(
        "Go chill on the La-Z-boy.", [
            "La-Z-boy, furniture",
        ])
    match_reference_entities(
        "The art gallery displayed some exquisite Frida Kahlo pieces.", [
            "Frida Kahlo, person",
            "art gallery, place",
            "pieces, artwork",
            "pieces, concept",
            "pieces, media",
        ])
    match_reference_entities(
        "Hide the painting under the stairs.", [
            "painting, artwork",
            "painting, media",
            "stairs, structure",
        ])


def test_entity_classifier_case_future_date_or_time():
    """
    Future date or time

    :return:
    """
    match_reference_entities(
        "She plans to retire after March 15th next year.", [
            "March 15th next year, date or time",
        ])
    match_reference_entities(
        "The train comes at 3 PM.", [
            "3 PM, date or time",
        ])


def test_entity_classifier_case_game_or_playful_activity():
    """
    Plant or flora.

    :return:
    """
    match_reference_entities(
        "Chess and Go are about strategy and foresight.", [
            "Chess, game",
            "Go, game",
            "foresight, concept",
            "strategy, concept",
        ])
    match_reference_entities(
        "Poker has some strategy but is more about reading people and being unreadable to others.", [
            "Poker, game",
            "being unreadable, activity",
            "being unreadable, capability",
            "being unreadable, concept",
            "reading people, activity",
            "reading people, capability",
            "reading people, concept",
            "strategy, concept",
        ])
    match_reference_entities(
        "The level of skill in the NBA has become so phenomenal as each generation of athletes push the bar higher.", [
            "NBA, organization",
            "athletes, person",
            "bar, concept",
            "generation, concept",
            "skill, capability",
            "skill, concept",
        ])


def test_entity_classifier_case_government_organization():
    """
    Government organization

    :return:
    """
    match_reference_entities(
        "NASA is responsible for the U.S. space program.", [
            "NASA, organization",
            "U.S. space program, concept",
            "U.S. space program, initiative or objective",
        ])
    match_reference_entities(
        "After the hurricane, FEMA was onsite to help with the recovery.", [
            "hurricane, event",
            "hurricane, phenomenon",
            "FEMA, organization",
            "recovery, activity",
            "recovery, concept",
            "recovery, initiative or objective",
        ])


def test_entity_classifier_case_government_or_other_social_program():
    """
    Government program

    :return:
    """
    match_reference_entities(
        "Meals on wheels is critical for the seniors who depend on it.", [
            "Meals on Wheels, organization",
            "Meals on wheels, organization",
            "seniors, person",
        ])
    match_reference_entities(
        "Social Security is an important component of America's social safety net", [
            "America, area",
            "America, place",
            "Social Security, concept",
            "social safety net, concept",
        ])
    match_reference_entities(
        "Many eduction reforms have been controversial, No Child Left Behind and Whole Reading are examples.", [
            "No Child Left Behind, initiative or objective",
            "Whole Reading, initiative or objective",
            "education reforms, concept",
            "education reforms, initiative or objective",
        ])


def test_entity_classifier_case_humanoid_person_or_persona():
    """
    Humanoid person or persona

    :return:
    """
    match_reference_entities(
        "Albert Einstein was a brilliant physicist.", [
            "Albert Einstein, person",
            "physicist, job or profession",
        ])
    match_reference_entities(
        "Sherlock Holmes is known for his keen observation skills and his trusted partner, Dr. Watson.", [
            "Sherlock Holmes, person",
            "Dr. Watson, person",
            "observation skills, capability",
        ])


def test_entity_classifier_case_interpersonal_or_relational_concept():
    """
    Interpersonal or relational concept

    :return:
    """
    match_reference_entities(
        "Trust is the foundation of a strong relationship and effective communication is how we build it.", [
            "Trust, concept",
            "communication, concept",
            "relationship, concept",
        ])


def test_entity_classifier_case_job_trade_or_profession():
    """
    Job, trade, career, or profession

    :return:
    """
    match_reference_entities(
        "With the shift towards high-tech, traditional trades like carpentry have become comparatively rare.", [
            "carpentry, activity",
            "carpentry, trade",
            "high-tech, concept",
            "traditional trades, concept",
        ])
    match_reference_entities(
        "Although his father got started in the Marines, he wouldn't be allowed follow the same path.", [
            "father, person",
            "Marines, organization",
        ])


def test_entity_classifier_case_legal_concept():
    """
    Material or substance

    :return:
    """
    match_reference_entities(
        ("The rule of law is based on the fundamental principle that \"right is right\", "
         "rather than \"might is right\"."), [
            "fundamental principle, concept",
            "might is right, concept",
            "might, concept",
            "right is right, concept",
            "right, concept",
            "rule of law, concept",
        ])
    match_reference_entities(
        "A judge is a public servant that provides justice to people harmed by the actions of others.", [
            "actions, activity",
            "actions, concept",
            "judge, job or profession",
            "justice, concept",
            "people, person",
            "public servant, job or profession",
        ])
    match_reference_entities(
        "Judges must remain neutral and unbiased, treating all parties equally without favoritism or prejudice.", [
            "Judges, job or profession",
            "favoritism, concept",
            "neutral, trait",
            "parties, concept",
            "parties, organization",
            "prejudice, concept",
            "unbiased, trait",
        ])
    match_reference_entities(
        ("All prospective justices claim to respect stare decisis but whenever an actual chance to tip the scales is "
         "before them, we have to assume that each would vote based on conscience and principles."), [
            "conscience, concept",
            "justices, job or profession",
            "justices, person",
            "principles, concept",
            "stare decisis, concept",
        ])


def test_entity_classifier_case_material_or_substance():
    """
    Material or substance

    :return:
    """
    match_reference_entities(
        "The sculpture was crafted from a single block of marble.", [
            "block of marble, material or substance",
            "sculpture, structure",
        ])
    match_reference_entities(
        "The fabric felt soft and luxurious against her skin.", [
            "fabric, material or substance",
            "skin, body part",
        ])
    match_reference_entities(
        ("It is her job to open packages so she did, but she was clearly disturbed when a white powder ejected out of "
         "this particular container right when she opened it."), [
            "container, container",
            "job, job or profession",
            "packages, container",
            "white powder, material or substance",
        ])


def test_entity_classifier_case_musical_instrument():
    """
    Musical instrument

    :return:
    """
    match_reference_entities(
        "On the piano, he was like da Vinci with a paint brush.", [
            "da Vinci, person",
            "paint brush, instrument",
            "piano, instrument",
        ])
    match_reference_entities(
        "You should have seen how he made those people dance, playing that hacksaw like Yo-Yo Ma on the cello.", [
            "Yo-Yo Ma, person",
            "cello, instrument",
            "dance, activity",
            "hacksaw, instrument",
        ])
    match_reference_entities(
        "Ain't nobody plays the trumpet like Miles Davis.", [
            "Miles Davis, person",
            "trumpet, instrument",
        ])


def test_entity_classifier_case_natural_or_artificial_terrain():
    """
    Natural or artificial terrain feature

    :return:
    """
    match_reference_entities(
        "They were massacred after the cavalry chased them into a box canyon and rained death from above.", [
            "box canyon, place",
            "cavalry, organization",
            "death, concept",
            "death, phenomenon",
        ])
    match_reference_entities(
        "The lake created a natural barrier for the men in the castle, who relied on it's remoteness for protection.", [
            "castle, structure",
            "lake, area",
            "lake, place",
            "men, person",
            "protection, concept",
            "remoteness, concept",
        ])


def test_entity_classifier_case_natural_resource():
    """
    Natural resource

    :return:
    """
    match_reference_entities(
        "Crude oil is still a vital for many economies, even as the world moves toward renewables.", [
            "Crude oil, material or substance",
            "economies, concept",
            "renewables, concept",
        ])
    match_reference_entities(
        "The lithium deposits in the ancient lake bed could fuel a revolution in battery technology.", [
            "ancient lake bed, area",
            "ancient lake bed, place",
            "battery technology, concept",
            "lithium deposits, material or substance",
            "revolution in battery technology, concept",
            "revolution, concept",
            "revolution, phenomenon",
        ])
    match_reference_entities(
        ("Having used fertilizer containing sewage sludge, Bill was for forced to sell his family's land, "
         "which is now contaminated and longer useful for agriculture."), [
            "Bill, person",
            "agriculture, activity",
            "agriculture, concept",
            "contaminated land, area",
            "contaminated land, place",
            "contamination, phenomenon",
            "family's land, area",
            "family's land, place",
            "family, concept",
            "family, organization",
            "fertilizer, material or substance",
            "land, area",
            "sewage sludge, material or substance",
        ])


def test_entity_classifier_case_non_profit_industry_trace_or_professional_organization():
    """
    Non-profit industry, trade or professional organization

    :return:
    """
    match_reference_entities(
        "The American Medical Association sets standards for medical professionals.", [
            "American Medical Association, organization",
            "medical professionals, job or profession",
            "standards, concept",
        ])
    match_reference_entities(
        "The Teamsters are for politicians that support American workers.", [
            "American workers, concept",
            "American workers, person",
            "Teamsters, organization",
            "politicians, person",
        ])


def test_entity_classifier_case_non_profit_religious_cultural_or_community_organization():
    """
    Non-profit religious, cultural, or community organization

    :return:
    """
    match_reference_entities(
        "The Red Cross provides humanitarian aid worldwide.", [
            "Red Cross, organization",
            "humanitarian aid, concept",
            "worldwide, area",
        ])


def test_entity_classifier_case_past_date_or_time():
    """
    Past date or time

    :return:
    """
    match_reference_entities(
        "She says she'll remember that day in the summer of 1969 till her dying breath.", [
            "day, date or time",
            "summer of 1969, date or time",
        ])


def test_entity_classifier_case_park_or_nature_preserve():
    """
    Park or nature preserve.

    :return:
    """
    match_reference_entities(
        ("Yellowstone National Park is home to a variety of wildlife, including some that cause friction between "
         "the park and nearby farming communities."), [
            "Yellowstone National Park, place",
            "wildlife, fauna",
            "farming communities, organization",
        ])
    match_reference_entities(
        "Let's meet at the big playground at Lake Elizabeth.", [
            "playground, area",
            "playground, place",
            "Lake Elizabeth, place",
        ])
    match_reference_entities(
        ("Take your trailer there, where it's public land that belongs to the Park Service so you can camp "
         "and sleep in your vehicle."), [
            "Park Service, organization",
            "camp, activity",
            "public land, area",
            "sleep, activity",
            "trailer, vehicle",
            "vehicle, vehicle",
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
            "Norwegian King, person",
            "Scottish border horses, animal",
            "Scottish border horses, fauna",
            "Scottish border horses, vehicle",
            "Sámi tribes, organization",
            "archers, job or profession",
            "conscript, job or profession",
            "conscripts, person",
            "light spears, weapon",
            "nearby towns, place",
            "shield wall, activity",
            "shield wall, concept",
            "skirmishers, job or profession",
        ])
    match_reference_entities(
        ("The band vibed with the audience and with each song, amped up the energy, which you could see "
         "as the mosh pit circled with ever increasing ferocity."), [
            "audience, group",  # TODO: how do we connect a type like this the concept of a group?
            "audience, organization",
            "audience, person",
            "band, organization",
            "energy, concept",
            "ferocity, trait",
            "mosh pit, area",
            "song, media",
        ])


def test_entity_classifier_case_permanent_building_or_monument():
    """
    Permanent building or Monument.

    :return:
    """
    match_reference_entities(
        "The Eiffel Tower is one of the most recognizable landmarks in Paris.", [
            "Eiffel Tower, place",
            "Eiffel Tower, structure",
            "Paris, place",
        ])


def test_entity_classifier_case_plant_or_flora():
    """
    Plant or flora.

    :return:
    """
    match_reference_entities(
        ("The Amazon rainforest fosters a diverse fabric of species, like Victoria Amazonica (Giant Water Lily), "
         "Arecaceae (Palms), or Heliconia (Lobster Claw), each playing crucial roles in their ecosystem."), [
            "Amazon rainforest, area",
            "Arecaceae, flora",
            "Heliconia, flora",
            "Victoria Amazonica, flora",
            "ecosystem, concept",
        ])


def test_entity_classifier_case_political_concept():
    """
    Political concept

    :return:
    """
    match_reference_entities(
        "Democracy is based on the principle of equal representation.", [
            "Democracy, concept",
            "equal representation, concept",
        ])


def test_entity_classifier_case_psychological_concept():
    """
    Psychological concept

    :return:
    """
    match_reference_entities(
        "Cognitive dissonance occurs when beliefs and actions are inconsistent.", [
            "Cognitive dissonance, concept",
            "actions, concept",
            "beliefs, concept",
            "inconsistency, concept",
        ])


def test_entity_classifier_case_quantity_not_related_to_currency():
    """
    Quantity not related to currency

    :return:
    """
    match_reference_entities(
        "The recipe calls for two cups of flour.", [
            "flour, food",
            "two cups, quantity",
        ])


def test_entity_classifier_case_religious_concept():
    """
    Religious concept

    :return:
    """
    match_reference_entities(
        "Karma is a central concept in Hinduism and Buddhism.", [
            "Buddhism, concept",
            "Hinduism, concept",
            "Karma, concept",
        ])


def test_entity_classifier_case_ritual_or_tradition():
    """
    Ritual or tradition

    :return:
    """
    match_reference_entities(
        "Her heartfelt farewell brought tears to my eyes.", [
            "farewell, event",
            "tears, sensation",
        ])
    match_reference_entities(
        "As she walked past the memorial, she added a single flower to the other objects on display.", [
            "flower, flora",
            "memorial, structure",
            "objects, container",
        ])


def test_entity_classifier_case_scientific_or_technological_concept():
    """
    Scientific or technological concept

    :return:
    """
    match_reference_entities(
        "Quantum mechanics describes the behavior of particles at the atomic level.", [
            "Quantum mechanics, concept",
            "atomic level, area",
            "particles, fauna",
            "particles, phenomenon",
        ])
    match_reference_entities(
        "Tesla bet big on self-driving without LiDAR.", [
            "LiDAR, device",
            "LiDAR, technology",
            "Tesla, organization",
            "self-driving, capability",
        ])
    match_reference_entities(
        "The new model boasts an impressive battery life.", [
            "battery life, capability",
            "model, concept",
            "model, device",
        ])


def test_entity_classifier_case_smell_or_sensation():
    """
    Religious concept

    :return:
    """
    match_reference_entities(
        ("The aroma of freshly baked bread filled the whole kitchen, briefly transporting her back to a time and "
         "place long forgotten, back to her childhood home, to those summer afternoons, baking with her father."), [
            "aroma, sensation",
            "bread, food",
            "childhood home, place",
            "father, person",
            "freshly baked bread, food",
            "kitchen, place",
            "summer afternoons, date or time",
            "summer afternoons, time",
        ])
    match_reference_entities(
        ("The chocolate smelled so good and tasted so creamy and sweet that she did feel a little better, even though "
         "she also felt guilty for finding a little joy when it seemed like everything can only be sad."), [
            "chocolate, food",
            "joy, concept",
            "sadness, concept",
        ])
    match_reference_entities(
        ("The warm water sent a feeling of relief throughout her body, washing away the aches accumulated over many "
         "uncomfortable nights, sleeping in the wilderness."), [
            "aches, sensation",
            "feeling of relief, concept",
            "feeling of relief, sensation",
            "uncomfortable nights, duration",
            "warm water, sensation",
            "wilderness, place",
        ])
    match_reference_entities(
        "The aged oak of the antique table felt smooth when he brushed his hand against its surface.", [
            "aged oak, flora",
            "antique table, furniture",
            "oak, flora",
            "surface, concept",
            "surface, structure",
        ])


def test_entity_classifier_case_social_or_cultural_concept():
    """
    Religious concept

    :return:
    """
    match_reference_entities(
        "Cultural diversity and the free flow of people enriches societies.", [
            "Cultural diversity, concept",
            "free flow of people, concept",
            "societies, area",
            "societies, concept",
        ])


def test_entity_classifier_case_street_address_or_fictional_place():
    """
    Street address or fictional place.

    :return:
    """
    match_reference_entities(
        "She lives at 123 Maple Street.", [
            "123 Maple Street, place",
        ])
    match_reference_entities(
        "Make sure they see this at 10 Downing.", [
            "10 Downing, place",
        ])
    match_reference_entities(
        "Hogwarts School of Witchcraft and Wizardry is located in Scotland.", [
            "Hogwarts School of Witchcraft and Wizardry, organization",
            "Scotland, place",
        ])
    match_reference_entities(
        "Little did Dorthy know that she would, once again, be stuck in the Land of Oz.", [
            "Dorthy, person",
            "Land of Oz, place",
        ])


def test_entity_classifier_case_storage_container():
    """
    Storage container.

    :return:
    """
    match_reference_entities(
        "She pulled out a pen and quickly scribbled the new information into her rolodex.", [
            "pen, instrument",
            "rolodex, container",
        ])
    match_reference_entities(
        "The box was filled with old photographs and letters.", [
            "box, container",
            "letters, media",
            "photographs, media",
        ])
    match_reference_entities(
        "He had a jar for loose change.", [
            "jar, container",
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
            "next month, date or time",
        ])
    match_reference_entities(
        "The grand opening of the new museum is in a year.", [
            "a year, duration",
            "grand opening, event",
            "museum, place",
            "year, date or time",
        ])
    match_reference_entities(
        "The signing of the Declaration of Independence was a pivotal moment in history.", [
            "Declaration of Independence, event",
        ])
    match_reference_entities(
        "The Olympics are held every four years.", [
            "Olympics, event",
            "four years, duration",
        ])


def test_entity_classifier_case_temporary_structure():
    """
    Temporary structure.

    :return:
    """
    match_reference_entities(
        "The circus tent will set up in the outer field for the next three weeks.", [
            "circus tent, structure",
            "outer field, area",
            "three weeks, duration",
        ])
    match_reference_entities(
        "The construction crew erected scaffolding around the building.", [
            "building, structure",
            "construction crew, organization",
            "scaffolding, structure",
        ])


def test_entity_classifier_case_utensil_or_machinery():
    """
    Utensil, machinery, or other mechanical tool.

    :return:
    """
    match_reference_entities(
        "The can opener is in the top drawer.", [
            "can opener, instrument",
            "top drawer, container",
        ])
    match_reference_entities(
        "Don't hold your spoon like that.", [
            "spoon, container",
        ])
    match_reference_entities(
        "The microscope is essential for studying microorganisms.", [
            "microorganisms, fauna",
            "microscope, device",
            "microscope, instrument",
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
            "Tesla, organization",
            "Model Y, vehicle",
        ])


if __name__ == "__main__":
    from observability.logging import setup_logging
    setup_logging()

    #test_entity_classifier_case_abstract_ability_or_attribute()
    #test_entity_classifier_case_animal_or_non_humanoid()
    #test_entity_classifier_case_article_book_document_or_other_text()
    #test_entity_classifier_case_artistic_or_literary_concept()
    #test_entity_classifier_case_audio_image_video_or_other_media()
    #test_entity_classifier_case_city_county_and_other_localities()
    #test_entity_classifier_case_clothing_shoes_or_jewelry()
    #test_entity_classifier_case_comment_message_letter_or_communication_artifact()
    #test_entity_classifier_case_computer_phone_or_electronic_device()
    #test_entity_classifier_case_construction_or_industrial_input()
    test_entity_classifier_case_contains_code()
    #test_entity_classifier_case_crime_terror_or_paramilitary()
    #test_entity_classifier_case_currency()
    #test_entity_classifier_case_economic_concept()
    #test_entity_classifier_case_ethical_existential_philosophical_or_social_concept()
    #test_entity_classifier_case_executive_operational_or_managerial_concept()
    #test_entity_classifier_case_food_drink_or_other_perishable()
    #test_entity_classifier_case_for_profit_business()
    #test_entity_classifier_case_furniture_or_art()
    #test_entity_classifier_case_future_date_or_time()
    #test_entity_classifier_case_game_or_playful_activity()
    #test_entity_classifier_case_government_or_other_social_program()
    #test_entity_classifier_case_government_organization()
    #test_entity_classifier_case_humanoid_person_or_persona()
    #test_entity_classifier_case_interpersonal_or_relational_concept()
    #test_entity_classifier_case_job_trade_or_profession()
    #test_entity_classifier_case_legal_concept()
    #test_entity_classifier_case_material_or_substance()
    #test_entity_classifier_case_musical_instrument()
    #test_entity_classifier_case_natural_or_artificial_terrain()
    #test_entity_classifier_case_natural_resource()
    #test_entity_classifier_case_non_profit_industry_trace_or_professional_organization()
    #test_entity_classifier_case_non_profit_religious_cultural_or_community_organization()
    #test_entity_classifier_case_park_or_nature_preserve()
    #test_entity_classifier_case_past_date_or_time()
    #test_entity_classifier_case_people_group()
    #test_entity_classifier_case_permanent_building_or_monument()
    #test_entity_classifier_case_plant_or_flora()
    #test_entity_classifier_case_political_concept()
    #test_entity_classifier_case_psychological_concept()
    #test_entity_classifier_case_quantity_not_related_to_currency()
    #test_entity_classifier_case_religious_concept()
    #test_entity_classifier_case_ritual_or_tradition()
    #test_entity_classifier_case_scientific_or_technological_concept()
    #test_entity_classifier_case_smell_or_sensation()
    #test_entity_classifier_case_social_or_cultural_concept()
    #test_entity_classifier_case_storage_container()
    #test_entity_classifier_case_street_address_or_fictional_place()
    #test_entity_classifier_case_temporal_event()
    #test_entity_classifier_case_temporary_structure()
    #test_entity_classifier_case_utensil_or_machinery()
    #test_entity_classifier_case_vehicle()

