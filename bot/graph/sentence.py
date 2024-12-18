import json
from db.neo4j import Neo4jDriver

from bot.lang.utils import flair_text_feature_extraction, openai_text_feature_extraction
from observability.logging import logging, setup_logging

logger = logging.getLogger(__name__)


def add_sentence_to_graph(connection: Neo4jDriver, sentence: str):
    sentence = sentence.strip()
    sentence_flair_features = flair_text_feature_extraction(sentence)
    sentence_openai_features = json.loads(openai_text_feature_extraction(sentence))
    logger.info(f"sentence_flair_features: {sentence_flair_features}")
    logger.info(f"sentence_openai_features: {sentence_openai_features}")

    # Sentency
    sentence_record = connection.query("MERGE (s:Sentence {text: $sentence}) RETURN s",
                                       {"sentence": sentence})
    logger.info(f"sentence: {sentence_record}")

    # Emotion
    for emotion in sentence_openai_features["emotions"]:
        emotion_record = connection.query("MERGE (e:EmotionLabel {text: $emotion}) RETURN e",
                                          {"emotion": emotion})
        logger.info(f"emotion: {emotion_record}")
        emotion_link_record = connection.query(
            "MATCH (e:EmotionLabel {text: $emotion}), (s:Sentence {text: $sentence}) "
            "MERGE (e)-[r:CONTAINS]->(s) "
            "ON CREATE SET r.createdAt = timestamp(), r.intensity = $intensity, r.nuance = $nuance "
            "RETURN r", {
                "emotion": emotion, "intensity": sentence_openai_features["emotional_intensity"],
                "nuance": sentence_openai_features["emotional_nuance"],
                "sentence": sentence
            })
        logger.info(f"emotion link: {emotion_link_record}")

    # Entity
    for entity in sentence_openai_features["entities"]:
        entity_record = connection.query(
            "MERGE (e:Entity {text: $entity}) RETURN e",
            {"entity": entity["entity"]})
        logger.info(f"entity: {entity_record}")
        entity_link_record = connection.query(
            "MATCH (e:Entity {text: $entity}), (s:Sentence {text: $sentence}) "
            "MERGE (e)-[r:CONTAINS]->(s) "
            "ON CREATE SET r.createdAt = timestamp() "
            "RETURN r", {"entity": entity["entity"], "sentence": sentence})
        logger.info(f"entity link: {entity_link_record}")
        entity_type_record = connection.query(
            "MERGE (e:EntityType {text: $entity_type}) RETURN e",
            {"entity_type": entity["type"]})
        logger.info(f"entity type: {entity_type_record}")
        entity_type_link_record = connection.query(
            "MATCH (e:EntityType {text: $entity_type}), (s:Sentence {text: $sentence}) "
            "MERGE (e)-[r:CONTAINS]->(s) "
            "ON CREATE SET r.createdAt = timestamp() "
            "RETURN r", {"entity_type": entity["type"], "sentence": sentence})
        logger.info(f"entity type link: {entity_type_link_record}")

    # Knowledge category
    for knowledge_category in sentence_openai_features["knowledge"]:
        knowledge_category_record = connection.query("MERGE (k:KnowledgeCategory {text: $knowledge_category}) RETURN k",
                                                     {"knowledge_category": knowledge_category})
        logger.info(f"knowledge category: {knowledge_category_record}")
        knowledge_category_link_record = connection.query(
            "MATCH (k:KnowledgeCategory {text: $knowledge_category}), (s:Sentence {text: $sentence}) "
            "MERGE (k)-[r:CONTAINS]->(s) "
            "ON CREATE SET r.createdAt = timestamp() "
            "RETURN r", {"knowledge_category": knowledge_category, "sentence": sentence})
        logger.info(f"knowledge category link: {knowledge_category_link_record}")

    # Numerals, broken out from entities
    for numeral in sentence_openai_features["numerals"]:
        numeral_record = connection.query("MERGE (n:Numeral {text: $numeral}) RETURN n",
                                         {"numeral": numeral["value"]})
        logger.info(f"numeral: {numeral_record}")
        numeral_link_record = connection.query(
            "MATCH (n:Numeral {text: $numeral}), (s:Sentence {text: $sentence}) "
            "MERGE (n)-[r:CONTAINS]->(s) "
            "ON CREATE SET r.createdAt = timestamp() "
            "RETURN r", {"numeral": numeral["value"], "sentence": sentence})
        logger.info(f"numeral link: {numeral_link_record}")

        # Numeral type
        numeral_type_record = connection.query(
            "MERGE (n:NumeralType {text: $numeral_type}) RETURN n",
            {"numeral_type": numeral["type"]})
        logger.info(f"numeral type: {numeral_type_record}")
        numeral_type_link_record = connection.query(
            "MATCH (n:NumeralType {text: $numeral_type}), (s:Sentence {text: $sentence}) "
            "MERGE (n)-[r:CONTAINS]->(s) "
            "ON CREATE SET r.createdAt = timestamp() "
            "RETURN r", {"numeral_type": numeral["type"], "sentence": sentence})
        logger.info(f"numeral type link: {numeral_type_link_record}")

        # Numeral unit
        numeral_unit_record = connection.query(
            "MERGE (n:NumeralUnit {text: $numeral_unit}) RETURN n",
            {"numeral_unit": numeral["unit"]})
        logger.info(f"numeral unit: {numeral_unit_record}")
        numeral_unit_link_record = connection.query(
            "MATCH (n:NumeralUnit {text: $numeral_unit}), (s:Sentence {text: $sentence}) "
            "MERGE (n)-[r:CONTAINS]->(s) "
            "ON CREATE SET r.createdAt = timestamp() "
            "RETURN r", {"numeral_unit": numeral["unit"], "sentence": sentence})
        logger.info(f"numeral unit link: {numeral_unit_link_record}")

    # Sentiment, but maybe this should be an attribute on the sentence?
    sentiment = sentence_openai_features["sentiment"]
    sentiment_record = connection.query("MERGE (n:SentimentLabel {text: $sentiment}) RETURN n",
                                        {"sentiment": sentiment})
    logger.info(f"sentiment: {sentiment_record}")
    sentiment_link_record = connection.query(
        "MATCH (n:SentimentLabel {text: $sentiment}), (s:Sentence {text: $sentence}) "
        "MERGE (n)-[r:CONTAINS]->(s) "
        "ON CREATE SET r.createdAt = timestamp() "
        "RETURN r", {"sentiment": sentiment, "sentence": sentence})
    logger.info(f"sentiment link: {sentiment_link_record}")

    # Sentence subject
    for subject in sentence_openai_features["subjects"]:
        subject_record = connection.query("MERGE (t:SentenceSubject {text: $subject}) RETURN t",
                                        {"subject": subject})
        logger.info(f"subject: {subject_record}")
        subject_link_record = connection.query(
            "MATCH (t:SentenceSubject {text: $subject}), (s:Sentence {text: $sentence}) "
            "MERGE (t)-[r:CONTAINS]->(s) "
            "ON CREATE SET r.createdAt = timestamp() "
            "RETURN r", {"subject": subject, "sentence": sentence})
        logger.info(f"subject link: {subject_link_record}")

    # Topic
    for topic in sentence_openai_features["topics"]:
        topic_record = connection.query("MERGE (t:TopicLabel {text: $topic}) RETURN t",
                                        {"topic": topic})
        logger.info(f"topic: {topic_record}")
        topic_link_record = connection.query(
            "MATCH (n:TopicLabel {text: $topic}), (s:Sentence {text: $sentence}) "
            "MERGE (n)-[r:CONTAINS]->(s) "
            "ON CREATE SET r.createdAt = timestamp() "
            "RETURN r", {"topic": topic, "sentence": sentence})
        logger.info(f"topic link: {topic_link_record}")


def get_sentences_related_to_word(connection, word):
    query = """
    MATCH (w:Word {text: $word})-[:CONTAINS]->(s:Sentence)
    RETURN s.text AS sentence
    """
    return connection.query(query, {"word": word})


def get_sentences_related_to_any_of_the_words(connection, words):
    # Create a parameterized query to match any of the words
    query = """
    MATCH (w:Word)-[:CONTAINS]->(s:Sentence)
    WHERE w.text IN $words
    RETURN DISTINCT s.text AS sentence
    """
    return connection.query(query, {"words": words})


if __name__ == '__main__':
    import argparse

    setup_logging(global_level="INFO")

    parser = argparse.ArgumentParser(description='Think about an idea.')
    parser.add_argument("-d", "--delete", help='Delete all data.',
                        action="store_true", default=False)
    args = parser.parse_args()

    driver = Neo4jDriver()
    if args.delete:
        logger.info("deleting existing data")
        driver.delete_all_data()

    # Add sentences to the graph
    test_set = [
        "People are born and with time, they die.",
        "We laugh, love, and struggle to survive.",
        "In dark times, some turn inward, to family, or God.",
        "Sooner or later, all rest beneath the sod.",

        # Tests entity and numeral extraction
        ("In Revelation 13:18, it says, \"let the one who has understanding calculate the number of the beast, "
         "for it is the number of a man, and his number is 666\"."),
        "You can't break your 20 down the street because the 711 doesn't opens until 7am.",
        "Maybe you should call 1-800-222-1222, the poison control hotline, or even 911!",
    ]

    for test in test_set:
        add_sentence_to_graph(driver, test)

    # Query related sentences
    #related_sentences = get_sentences_related_to_word(driver, "cat")
    #for relation in related_sentences:
    #    print(relation['sentence'])

    driver.close()
