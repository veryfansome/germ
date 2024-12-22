import hashlib
import json
import signal
import threading
import time
from datetime import datetime, timedelta

from bot.lang.utils import (flair_text_feature_extraction, openai_detect_sentence_type,
                            extract_openai_emotion_features, extract_openai_entity_features,
                            openai_text_feature_extraction, split_to_sentences)
from bot.lang.examples import documents as doc_examples, sentences as sentence_examples
from db.neo4j import Neo4jDriver
from db.models import Sentence, SessionLocal
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
        logger.debug(f"document/time link: {time_record_link}")
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
        emotion_record = self.driver.query("MERGE (emotion:EmotionLabel {text: $emotion}) RETURN emotion",
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
        logger.debug(f"idea/time link: {time_record_link}")
        return idea_record

    def add_knowledge_category(self, knowledge_category: str):
        knowledge_category_record = self.driver.query("MERGE (k:KnowledgeCategory {text: $knowledge_category}) RETURN k",
                                                      {"knowledge_category": knowledge_category})
        logger.info(f"knowledge_category: {knowledge_category_record}")
        return knowledge_category_record

    def add_sentence(self, sentence: str,
                     current_rounded_time=None, flair_features=None, openai_features=None, sentence_node_type=None):
        sentence = sentence.strip()
        if current_rounded_time is None:
            current_rounded_time, _, _ = self.add_time()

        sentence_signature = hashlib.sha256(sentence.encode()).hexdigest()
        with SessionLocal() as session:
            sentence_rdb_record = session.query(
                Sentence).filter(Sentence.sentence_signature == sentence_signature).first()

            # Create record if not found
            if not sentence_rdb_record:
                sentence_rdb_record = Sentence(text=sentence, sentence_signature=sentence_signature)
                session.add(sentence_rdb_record)
                session.commit()

            # Get sentence type if not set
            if sentence_rdb_record.sentence_node_type is None:
                openai_sentence_type_result = ({"sentence_type": sentence_node_type}
                                               if sentence_node_type is not None
                                               else json.loads(openai_detect_sentence_type(sentence)))
                sentence_rdb_record.sentence_node_type = SENTENCE_NODE_TYPE[openai_sentence_type_result["sentence_type"]]
                session.commit()

            # Create sentence node and link to time
            sentence_node_type = sentence_rdb_record.sentence_node_type
            sentence_graph_record = self.driver.query(
                "MERGE (s:" + sentence_node_type + " {text: $text, sentence_id: $sentence_id}) RETURN s",
                {"text": sentence, "sentence_id": sentence_rdb_record.sentence_id})

            logger.info(f"sentence: {sentence_graph_record}")
            time_record_link = self.driver.query(
                "MATCH (t:Time {text: $time}), (s:" + sentence_node_type + " {text: $sentence}) "
                "MERGE (s)-[r:OCCURRED]->(t) "
                "RETURN r", {"time": str(current_rounded_time), "sentence": sentence})
            logger.debug(f"sentence/time link: {time_record_link}")

            # Create a linked idea because every sentence serializes at least one idea.
            self.add_idea(sentence, current_rounded_time=current_rounded_time)
            idea_link_record = self.driver.query(
                "MATCH (i:Idea {text: $idea}), (s:" + sentence_node_type + " {text: $sentence}) "
                "MERGE (s)<-[r:EXPRESSES]-(i) "
                "RETURN r", {"idea": sentence, "sentence": sentence})
            logger.info(f"idea link: {idea_link_record}")

            if sentence_rdb_record.sentence_openai_emotion_features is None:
                sentence_rdb_record.sentence_openai_emotion_features = json.loads(
                    extract_openai_emotion_features(sentence))
                session.commit()

            # Emotion
            openai_emotions_features = sentence_rdb_record.sentence_openai_emotion_features
            for emotion in openai_emotions_features["emotions"]:
                logger.info(f"emotion: {emotion}")
                self.add_emotion(emotion["emotion"])
                self.link_emotion_to_sentence(emotion["emotion"], sentence_rdb_record.sentence_id, sentence_node_type,
                                              intensity=emotion["intensity"],
                                              nuance=emotion["nuance"])

                # Add emotion source and target to entities
                self.add_entity(emotion["emotion_source"])
                self.link_entity_to_sentence(emotion["emotion_source"], sentence_rdb_record.sentence_id, sentence_node_type)
                self.add_entity_type(emotion["emotion_source_entity_type"])
                self.link_entity_type_to_entity(emotion["emotion_source"], emotion["emotion_source_entity_type"])

                self.add_entity(emotion["emotion_target"])
                self.link_entity_to_sentence(emotion["emotion_target"], sentence_rdb_record.sentence_id, sentence_node_type)
                self.add_entity_type(emotion["emotion_target_entity_type"])
                self.link_entity_type_to_entity(emotion["emotion_target"], emotion["emotion_target_entity_type"])

                for synonymous_emotion in emotion["synonymous_emotions"]:
                    self.add_emotion(synonymous_emotion)
                    self.link_emotion_to_emotion(emotion["emotion"], synonymous_emotion)

            if sentence_rdb_record.sentence_flair_text_features is None:
                sentence_rdb_record.sentence_flair_text_features = (flair_features if flair_features is not None
                                                                    else flair_text_feature_extraction(sentence))
                session.commit()
            flair_features = sentence_rdb_record.sentence_flair_text_features

            if sentence_rdb_record.sentence_openai_entity_features is None:
                sentence_rdb_record.sentence_openai_entity_features = json.loads(
                    extract_openai_entity_features(sentence))
                session.commit()
            openai_entity_features = sentence_rdb_record.sentence_openai_entity_features

            # Entity
            for proper_noun in flair_features["proper_nouns"]:
                self.add_entity(proper_noun)
                self.link_entity_to_sentence(proper_noun, sentence_rdb_record.sentence_id, sentence_node_type)
            for entity in flair_features["ner"]:
                self.add_entity(entity)
                self.link_entity_to_sentence(entity, sentence_rdb_record.sentence_id, sentence_node_type)
            for entity in openai_entity_features["entities"]:
                self.add_entity(entity["entity"])
                self.link_entity_to_sentence(entity["entity"], sentence_rdb_record.sentence_id, sentence_node_type)
                self.add_entity_type(entity["entity_type"])
                self.link_entity_type_to_entity(entity["entity"], entity["entity_type"])
                self.link_entity_to_sentiment(entity["entity"], entity["sentiment"])

            if sentence_rdb_record.sentence_openai_text_features is None:
                sentence_rdb_record.sentence_openai_text_features = (
                    openai_features if openai_features is not None
                    else json.loads(openai_text_feature_extraction(sentence)))
                session.commit()
            openai_features = sentence_rdb_record.sentence_openai_text_features

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

            # Verb
            #for verb in flair_features["verbs"]:
            #    # TODO: Filter out is, are, were, and any other states of being verbs.
            #    #       Instead of saving them as entities, I should save them as relationships between the entities.
            #    #       It's important that this graph not just be a graph of words and ideas but also model out the ideas
            #    #       in the actual relationships of the entities as well.
            #    self.add_verb(verb)
            #    self.link_verb_to_sentence(verb, sentence, sentence_node_type)

            return sentence_graph_record, sentence_node_type

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
        logger.debug(f"time links: {time_record_links}")
        return current_rounded_time, current_time_record, time_record_links

    def add_topic_label(self, topic: str):
        topic_record = self.driver.query("MERGE (e:TopicLabel {text: $topic}) RETURN e",
                                         {"topic": topic})
        logger.info(f"topic_label: {topic_record}")
        return topic_record

    def add_verb(self, verb: str):
        verb_record = self.driver.query("MERGE (v:Verb {text: $verb}) RETURN v",
                                        {"verb": verb})
        logger.info(f"verb: {verb_record}")
        return verb_record

    def close(self):
        self.driver.close()

    def delete_all_data(self):
        self.driver.delete_all_data()

    def get_similar_but_disconnected_ideas_by_random_topic(self):
        # Get a random topic that is linked to two ideas that are not the same and not connected
        topic_results = self.driver.query(
            "MATCH (idea1:Idea)-[:EXPRESSES]->(s1:DeclarativeSentence)<-[:CONTAINS]-(topic:TopicLabel) "
            "MATCH (idea2:Idea)-[:EXPRESSES]->(s2:DeclarativeSentence)<-[:CONTAINS]-(topic) "
            "WHERE idea1 <> idea2 "
            "AND NOT (idea1)-[:RELATED]-(idea2) "
            "ORDER BY rand() "
            f"LIMIT 1 "
            "RETURN topic")
        # Get ideas that are linked by this topic
        if topic_results:
            logger.info(f"random topic: {topic_results[0]['topic']['text']}")
            idea_results = self.driver.query(
                "MATCH (idea1:Idea)-[:EXPRESSES]->(s1:DeclarativeSentence)<-[:CONTAINS]-(t:TopicLabel {text: $topic}) "
                "MATCH (idea2:Idea)-[:EXPRESSES]->(s2:DeclarativeSentence)<-[:CONTAINS]-(t) "
                "WHERE idea1 <> idea2 "
                "AND NOT (idea1)-[:RELATED]-(idea2) "
                "AND idea1.text < idea2.text "  # Sorting ensures unique combo
                "RETURN DISTINCT idea1, idea2", {"topic": topic_results[0]["topic"]["text"]})
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

    def link_emotion_to_emotion(self, emotion1: str, emotion2_id: int):
        emotion_link_record = self.driver.query(
            "MATCH (e1:EmotionLabel {text: $emotion1}), (e2:EmotionLabel {text: $emotion2}) "
            "MERGE (e1)-[r:RELATED]->(e2) "
            "RETURN r", {"emotion1": emotion1, "emotion2": emotion2_id})
        logger.info(f"emotion/emotion link: {emotion_link_record}")
        return emotion_link_record

    def link_emotion_to_sentence(self, emotion: str, sentence_id: int, sentence_node_type: str,
                                 intensity: str = "none", nuance: str = "none"):
        logger.info(f"link_emotion_to_sentence: {emotion} {sentence_id} {sentence_node_type} {intensity} {nuance}")
        emotion_link_record = self.driver.query(
            "MATCH (e:EmotionLabel {text: $emotion}), (s:" + sentence_node_type + " {sentence_id: $sentence_id}) "
            "MERGE (e)-[r:CONTAINS]->(s) "
            "ON CREATE SET r.intensity = $intensity, r.nuance = $nuance "
            "RETURN r", {
                "emotion": emotion, "sentence_id": sentence_id,
                "intensity": intensity, "nuance": nuance,
            })
        logger.info(f"emotion link: {emotion_link_record}")
        return emotion_link_record

    def link_entity_to_sentence(self, entity: str, sentence_id: int, sentence_node_type: str):
        entity_link_record = self.driver.query(
            "MATCH (e:Entity {text: $entity}), (s:" + sentence_node_type + " {sentence_id: $sentence_id}) "
            "MERGE (e)-[r:CONTAINS]->(s) "
            "RETURN r", {"entity": entity, "sentence_id": sentence_id})
        logger.info(f"entity link: {entity_link_record}")
        return entity_link_record

    def link_entity_to_sentiment(self, entity: str, sentiment: str):
        entity_link_record = self.driver.query(
            "MATCH (e:Entity {text: $entity}), (s:SentimentLabel {text: $sentiment}) "
            "MERGE (e)-[r:VIBED]-(s) "
            "RETURN r", {"entity": entity, "sentiment": sentiment})
        logger.info(f"entity/sentiment link: {entity_link_record}")
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
            "MERGE (n)-[r:VIBED]-(s) "
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

    def link_verb_to_sentence(self, verb: str, sentence: str, sentence_node_type: str):
        verb_link_record = self.driver.query(
            "MATCH (v:Verb {text: $verb}), (s:" + sentence_node_type + " {text: $sentence}) "
            "MERGE (v)-[r:CONTAINS]->(s) "
            "RETURN r", {"verb": verb, "sentence": sentence})
        logger.info(f"verb link: {verb_link_record}")
        return verb_link_record

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
