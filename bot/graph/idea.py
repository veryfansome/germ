import json
import signal
import threading
import time
from datetime import datetime, timedelta
from db.neo4j import Neo4jDriver

from bot.lang.utils import flair_text_feature_extraction, openai_text_feature_extraction, split_to_sentences
from bot.lang.examples import documents as doc_examples, sentences as sentence_examples
from observability.logging import logging, setup_logging

logger = logging.getLogger(__name__)

lock = threading.RLock()

INSTANCE_MANIFEST = {}
SENTENCE_NODE_TYPE = {
    "complex": "ComplexSentence",
    "conditional": "ConditionalSentence",
    "declarative": "DeclarativeSentence",
    "exclamatory": "ExclamatorySentence",
    "imperative": "ImperativeSentence",
    "interrogative": "InterrogativeSentence",
}


class IdeaGraph:
    def __init__(self, driver: Neo4jDriver):
        self.driver = driver

    def add_document(self, name: str, body: str):
        """
        An article, a poem, or even a chapter of a book.

        :param name:
        :param body:
        :return:
        """
        current_rounded_time, _, _ = self.add_time()
        document_record = self.driver.query("MERGE (d:Document {name: $name}) RETURN d",
                                            {"name": name})
        logger.info(f"document: {document_record}")
        time_record_link = self.driver.query(
            "MATCH (t:Time {text: $time}), (d:Document {name: $name}) "
            "MERGE (d)-[r:OCCURRED]-(t) "
            "RETURN r", {"time": str(current_rounded_time), "name": name})
        logger.info(f"document/time link: {time_record_link}")
        for body_sentence in split_to_sentences(body):
            _, body_sentence_node_type = self.add_sentence(body_sentence)
            body_sentence_link_record = self.driver.query(
                "MATCH (d:Document {name: $document}), (s:" + body_sentence_node_type + " {text: $sentence}) "
                "MERGE (s)-[r:CONTAINS]->(d) "
                "RETURN r", {
                    "document": name, "sentence": body_sentence,
                })
            logger.info(f"sentence link: {body_sentence_link_record}")
        return document_record

    def add_emotion(self, emotion: str):
        emotion_record = self.driver.query("MERGE (e:EmotionLabel {text: $emotion}) RETURN e",
                                          {"emotion": emotion})
        logger.info(f"emotion: {emotion_record}")
        return emotion_record

    def add_entity(self, entity: str):
        entity_record = self.driver.query("MERGE (e:Entity {text: $entity}) RETURN e",
                                          {"entity": entity})
        logger.info(f"entity: {entity_record}")
        return entity_record

    def add_entity_type(self, entity_type: str):
        entity_type_record = self.driver.query("MERGE (e:EntityType {text: $entity_type}) RETURN e",
                                               {"entity_type": entity_type})
        logger.info(f"entity_type: {entity_type_record}")
        return entity_type_record

    def add_idea(self, idea: str, current_rounded_time=None):
        if current_rounded_time is None:
            current_rounded_time, _, _ = self.add_time()
        idea_record = self.driver.query("MERGE (i:Idea {text: $idea}) RETURN i", {"idea": idea})
        logger.info(f"idea: {idea_record}")
        time_record_link = self.driver.query(
            "MATCH (t:Time {text: $time}), (i:Idea {text: $idea}) "
            "MERGE (i)-[r:OCCURRED]->(t) "
            "RETURN r", {"time": str(current_rounded_time), "idea": idea})
        logger.info(f"idea/time link: {time_record_link}")
        return idea_record

    def add_knowledge_category(self, knowledge_category: str):
        knowledge_category_record = self.driver.query("MERGE (k:KnowledgeCategory {text: $knowledge_category}) RETURN k",
                                                      {"knowledge_category": knowledge_category})
        logger.info(f"knowledge_category: {knowledge_category_record}")
        return knowledge_category_record

    def add_sentence(self, sentence: str, current_rounded_time=None, flair_features=None, openai_features=None):
        sentence = sentence.strip()

        # TODO: Integrate PostgreSQL calls here so we can store and reuse text features
        # TODO: Flesh out verbs and other flair stuff

        flair_features = flair_features if flair_features is not None else flair_text_feature_extraction(sentence)
        openai_features = openai_features if openai_features is not None else json.loads(openai_text_feature_extraction(sentence))
        sentence_node_type = SENTENCE_NODE_TYPE[openai_features["sentence_type"]]

        if current_rounded_time is None:
            current_rounded_time, _, _ = self.add_time()

        # Sentence
        sentence_record = self.driver.query("MERGE (s:" + sentence_node_type + " {text: $sentence}) RETURN s",
                                            {"sentence": sentence})
        logger.info(f"sentence: {sentence_record}")
        time_record_link = self.driver.query(
            "MATCH (t:Time {text: $time}), (s:" + sentence_node_type + " {text: $sentence}) "
            "MERGE (s)-[r:OCCURRED]->(t) "
            "RETURN r", {"time": str(current_rounded_time), "sentence": sentence})
        logger.info(f"sentence/time link: {time_record_link}")

        # Create a linked idea because every sentence serializes at least one idea.
        self.add_idea(sentence, current_rounded_time=current_rounded_time)
        idea_link_record = self.driver.query(
            "MATCH (i:Idea {text: $idea}), (s:" + sentence_node_type + " {text: $sentence}) "
            "MERGE (s)<-[r:EXPRESSES]-(i) "
            "RETURN r", {"idea": sentence, "sentence": sentence})
        logger.info(f"idea link: {idea_link_record}")

        # Emotion
        for emotion in openai_features["emotions"]:
            self.add_emotion(emotion)
            self.link_emotion_to_sentence(emotion, sentence, sentence_node_type,
                                          intensity=openai_features["emotional_intensity"],
                                          nuance=openai_features["emotional_nuance"])

        # Entity
        for entity in openai_features["entities"]:
            self.add_entity(entity["entity"])
            self.link_entity_to_sentence(entity["entity"], sentence, sentence_node_type)
            self.add_entity_type(entity["type"])
            self.link_entity_type_to_entity(entity["entity"], entity["type"])

        # Knowledge category
        for knowledge_category in openai_features["knowledge"]:
            self.add_knowledge_category(knowledge_category)
            self.link_knowledge_category_to_sentence(knowledge_category, sentence, sentence_node_type)

        # Sentiment, but maybe this should be an attribute on the sentence?
        sentiment = openai_features["sentiment"]
        self.add_sentiment_label(sentiment)
        self.link_sentiment_to_sentence(sentiment, sentence, sentence_node_type)

        # Sentence subject
        for subject in openai_features["subjects"]:
            self.add_sentence_subject(subject)
            self.link_sentence_subject_to_sentence(subject, sentence, sentence_node_type)

        # Topic
        for topic in openai_features["topics"]:
            self.add_topic_label(topic)
            self.link_topic_label_to_sentence(topic, sentence, sentence_node_type)

        return sentence_record, sentence_node_type

    def add_sentence_subject(self, subject: str):
        subject_record = self.driver.query("MERGE (n:SentenceSubject {text: $subject}) RETURN n",
                                           {"subject": subject})
        logger.info(f"sentence_subject: {subject_record}")
        return subject_record

    def add_sentiment_label(self, sentiment: str):
        sentiment_record = self.driver.query("MERGE (n:SentimentLabel {text: $sentiment}) RETURN n",
                                             {"sentiment": sentiment})
        logger.info(f"sentiment_label: {sentiment_record}")
        return sentiment_record

    def add_time(self):
        """
        Time nodes are created in set intervals and changed together to match the directionality of real time.
        The idea, here, is that every node created should be linked to its proper chunk of time. This provides a sense
        of temporality.

        :return:
        """
        now = datetime.now().replace(second=0, microsecond=0)
        current_rounded_time = round_time_now_down_to_nearst_15(now)
        # This lock throttles Neo4j calls but is a lazy way to make sure Time nodes are not created more than once.
        # Even with the `MERGE`, if multiple identical such queries fire concurrently, duplicates are created.
        with lock:
            current_time_record = self.driver.query("MERGE (t:Time {text: $time, epoch: $epoch}) RETURN t",
                                                    {"time": str(current_rounded_time), "epoch": int(time.time())})
        logger.info(f"time: {current_time_record}")
        last_rounded_time = current_rounded_time - timedelta(minutes=15)
        time_record_links = self.driver.query(
            "MATCH (n:Time {text: $now}), (l:Time {text: $last}) "
            "MERGE (n)-[rp:PRECEDES]->(l) "
            "MERGE (l)-[rf:FOLLOWS]->(n) "
            "RETURN rp, rf", {"now": str(current_rounded_time), "last": str(last_rounded_time)})
        logger.info(f"time links: {time_record_links}")
        return current_rounded_time, current_time_record, time_record_links

    def add_topic_label(self, topic: str):
        topic_record = self.driver.query("MERGE (e:TopicLabel {text: $topic}) RETURN e",
                                         {"topic": topic})
        logger.info(f"topic_label: {topic_record}")
        return topic_record

    def close(self):
        self.driver.close()

    def delete_all_data(self):
        self.driver.delete_all_data()

    def get_similar_but_disconnected_ideas_by_random_topic(self):
        # Get a random topic that is linked to two ideas that are not the same and not connected
        topic_results = self.driver.query(
            "MATCH (i1:Idea)-[:EXPRESSES]->(s1:DeclarativeSentence)<-[:CONTAINS]-(topic:TopicLabel) "
            "MATCH (i2:Idea)-[:EXPRESSES]->(s2:DeclarativeSentence)<-[:CONTAINS]-(topic) "
            "WHERE i1 <> i2 "
            "AND NOT (i1)-[:RELATED]-(i2) "
            "ORDER BY rand() "
            f"LIMIT 1 "
            "RETURN topic")
        # Get ideas that are linked by this topic
        if topic_results:
            logger.info(f"random topic: {topic_results[0]['topic']['text']}")
            idea_results = self.driver.query(
                "MATCH (i1:Idea)-[:EXPRESSES]->(s1:DeclarativeSentence)<-[:CONTAINS]-(t:TopicLabel {text: $topic}) "
                "MATCH (i2:Idea)-[:EXPRESSES]->(s2:DeclarativeSentence)<-[:CONTAINS]-(t) "
                "WHERE i1 <> i2 "
                "AND NOT (i1)-[:RELATED]-(i2) "
                "AND i1.text < i2.text "  # Sorting ensures unique combo
                "RETURN DISTINCT i1, i2", {"topic": topic_results[0]["topic"]["text"]})
            return topic_results, idea_results
        return topic_results, []

    def get_random_ideas(self, count: int = 2):
        results = self.driver.query(
            "MATCH (idea:Idea) "
            "WITH idea "
            "ORDER BY rand() "
            f"LIMIT {count} "
            "RETURN idea")
        return results

    def link_emotion_to_sentence(self, emotion: str, sentence: str, sentence_node_type: str,
                                 intensity: str = None, nuance: str = None):
        emotion_link_record = self.driver.query(
            "MATCH (e:EmotionLabel {text: $emotion}), (s:" + sentence_node_type + " {text: $sentence}) "
            "MERGE (e)-[r:CONTAINS]->(s) "
            "ON CREATE SET r.createdAt = timestamp(), r.intensity = $intensity, r.nuance = $nuance "
            "ON MATCH SET r.lastMatchedAt = timestamp() "
            "RETURN r", {
                "emotion": emotion, "sentence": sentence,
                "intensity": intensity, "nuance": nuance,
            })
        logger.info(f"emotion link: {emotion_link_record}")
        return emotion_link_record

    def link_entity_to_sentence(self, entity: str, sentence: str, sentence_node_type: str):
        entity_link_record = self.driver.query(
            "MATCH (e:Entity {text: $entity}), (s:" + sentence_node_type + " {text: $sentence}) "
            "MERGE (e)-[r:CONTAINS]->(s) "
            "RETURN r", {"entity": entity, "sentence": sentence})
        logger.info(f"entity link: {entity_link_record}")
        return entity_link_record

    def link_entity_type_to_entity(self, entity: str, entity_type: str):
        entity_type_link_record = self.driver.query(
            "MATCH (e:Entity {text: $entity}), (t:EntityType {text: $entity_type}) "
            "MERGE (t)-[r:CONTAINS]->(e) "
            "RETURN r", {"entity": entity, "entity_type": entity_type})
        logger.info(f"entity type link: {entity_type_link_record}")
        return entity_type_link_record

    def link_knowledge_category_to_sentence(self, knowledge_category: str, sentence: str, sentence_node_type: str):
        knowledge_category_link_record = self.driver.query(
            "MATCH (k:KnowledgeCategory {text: $knowledge_category}), (s:" + sentence_node_type + " {text: $sentence}) "
            "MERGE (k)-[r:CONTAINS]->(s) "
            "RETURN r", {"knowledge_category": knowledge_category, "sentence": sentence})
        logger.info(f"knowledge category link: {knowledge_category_link_record}")

    def link_sentiment_to_sentence(self, sentiment: str, sentence: str, sentence_node_type: str):
        sentiment_link_record = self.driver.query(
            "MATCH (n:SentimentLabel {text: $sentiment}), (s:" + sentence_node_type + " {text: $sentence}) "
            "MERGE (n)-[r:CONTAINS]->(s) "
            "RETURN r", {"sentiment": sentiment, "sentence": sentence})
        logger.info(f"sentiment link: {sentiment_link_record}")
        return sentiment_link_record

    def link_sentence_subject_to_sentence(self, sentence_subject: str, sentence: str, sentence_node_type: str):
        sentence_subject_link_record = self.driver.query(
            "MATCH (n:SentenceSubject {text: $sentence_subject}), (s:" + sentence_node_type + " {text: $sentence}) "
            "MERGE (n)-[r:CONTAINS]->(s) "
            "RETURN r", {"sentence_subject": sentence_subject, "sentence": sentence})
        logger.info(f"sentence_subject link: {sentence_subject_link_record}")
        return sentence_subject_link_record

    def link_topic_label_to_sentence(self, topic_label: str, sentence: str, sentence_node_type: str):
        topic_label_link_record = self.driver.query(
            "MATCH (t:TopicLabel {text: $topic_label}), (s:" + sentence_node_type + " {text: $sentence}) "
            "MERGE (t)-[r:CONTAINS]->(s) "
            "RETURN r", {"topic_label": topic_label, "sentence": sentence})
        logger.info(f"topic_label link: {topic_label_link_record}")
        return topic_label_link_record

    def merge_ideas(self, idea_to_keep: str, idea_to_merge: str):
        results = self.driver.query("""
MATCH (idea_to_keep:Idea {text: $idea_to_keep}), (idea_to_merge:Idea {text: $idea_to_merge})
WITH idea_to_keep, idea_to_merge
MATCH (idea_to_merge)-[r:EXPRESSES]->(x)
MERGE (idea_to_keep)-[:EXPRESSES]->(x)
DELETE r
WITH idea_to_keep, idea_to_merge
MATCH (idea_to_merge)-[r:OCCURRED]->(x)
DELETE r
DELETE idea_to_merge
RETURN idea_to_keep
""".strip(), {
                "idea_to_keep": idea_to_keep, "idea_to_merge": idea_to_merge})
        logger.info(f"idea kept: {results}")

    def signal_handler(self, sig, frame):
        self.close()


def get_idea_graph(name: str):
    if name not in INSTANCE_MANIFEST:
        new_instance = IdeaGraph(Neo4jDriver())
        signal.signal(signal.SIGINT, new_instance.signal_handler)
        INSTANCE_MANIFEST[name] = new_instance
    return INSTANCE_MANIFEST[name]


def round_time_now_down_to_nearst_15(dt):
    minutes = dt.minute
    remainder = minutes % 15  # How many 15-minute increments have passed
    rounded_minutes = minutes - remainder  # Subtract to round down
    return dt.replace(minute=rounded_minutes, second=0, microsecond=0)


if __name__ == '__main__':
    import argparse

    setup_logging(global_level="INFO")

    parser = argparse.ArgumentParser(description='Build a graph of ideas.')
    parser.add_argument("--load-examples", action="store_true", help='Load example data.',
                        default=False)
    args = parser.parse_args()
    idea_graph = get_idea_graph(__name__)

    if args.load_examples:
        for doc_title, doc_body in doc_examples.all_examples:
            idea_graph.add_document(doc_title, doc_body)

        # Add sentences to the graph
        for exp in sentence_examples.all_examples:
            idea_graph.add_sentence(exp)

    idea_graph.close()
