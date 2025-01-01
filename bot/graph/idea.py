import asyncio
import json
import re
import signal
import uuid
from abc import ABC, abstractmethod
from asyncio import Task
from datetime import datetime, timedelta, timezone
from sqlalchemy.future import select as sql_select
from starlette.concurrency import run_in_threadpool

from bot.graph.entities import default_entity_types
from bot.lang.classifiers import (emotion_to_entity_classifier,
                                  extract_openai_text_features, get_entity_type_classifier,
                                  sentence_classifier, split_to_sentences)
from bot.lang.examples import documents as doc_examples, sentences as sentence_examples
from db.neo4j import AsyncNeo4jDriver
from db.models import AsyncSessionLocal, Sentence
from observability.logging import logging, setup_logging
from settings.germ_settings import UUID5_NS

logger = logging.getLogger(__name__)

SENTENCE_NODE_TYPE = {
    "complex": "ComplexSentence",
    "conditional": "ConditionalSentence",
    "declarative": "DeclarativeSentence",
    "exclamatory": "ExclamatorySentence",
    "imperative": "ImperativeSentence",
    "interrogative": "InterrogativeSentence",
}


class DeclarativeSentenceMergeEventHandler(ABC):
    @abstractmethod
    async def on_merge(self, node_type: str, sentence_id: int, openai_parameters):
        pass


class IdeaGraph:
    def __init__(self, driver: AsyncNeo4jDriver):
        self.driver = driver
        self.declarative_sentence_merge_event_handlers: list[DeclarativeSentenceMergeEventHandler] = []

    async def add_default_entity_types(self) -> list[Task]:
        async_tasks = []
        for entity_type in default_entity_types:
            async_tasks.append(asyncio.create_task(self.add_entity_type(entity_type)))
        return async_tasks

    async def add_document(self, name: str, body: str):
        """
        An article, a poem, or even a chapter of a book.

        :param name:
        :param body:
        :return:
        """
        document_record = await self.driver.query(
            "MERGE (d:Document {name: $name}) RETURN d", {"name": name})
        logger.info(f"document: {document_record}")

        last_sentence_id = None
        last_sentence_node_type = None
        for body_sentence in split_to_sentences(body):
            _, this_sentence_node_type, this_sentence_id, async_tasks = await self.add_sentence(body_sentence)

            sentence_do_document_link_record = await self.driver.query(
                "MATCH (d:Document {name: $document}), (s:" + this_sentence_node_type + " {sentence_id: $sentence_id}) "
                "MERGE (d)-[r:CONTAINS]->(s) "
                "RETURN r", {
                    "document": name, "sentence_id": this_sentence_id
                })
            if last_sentence_id is not None:
                sentence_to_sentence_link = await self.driver.query(
                    "MATCH (l:" + last_sentence_node_type + " {sentence_id: $last_sentence_id}), (t:" + this_sentence_node_type + " {sentence_id: $this_sentence_id}) "
                    "MERGE (l)-[r:PRECEDES]->(t) "
                    "RETURN r", {
                        "last_sentence_id": last_sentence_id, "this_sentence_id": this_sentence_id,
                    })
                logger.info(f"sentence/sentence link: {sentence_to_sentence_link}")

            logger.info(f"sentence link: {sentence_do_document_link_record}")
            last_sentence_id = this_sentence_id
            last_sentence_node_type = this_sentence_node_type
        return document_record

    async def add_emotion(self, emotion: str):
        emotion_record = await self.driver.query(
            "MERGE (emotion:EmotionLabel {text: $emotion}) RETURN emotion", {"emotion": emotion})
        logger.info(f"emotion: {emotion_record}")
        return emotion_record

    async def add_entity(self, entity: str):
        entity_record = await self.driver.query(
            "MERGE (e:Entity {text: $entity}) RETURN e", {"entity": entity})
        logger.info(f"entity: {entity_record}")
        return entity_record

    async def add_entity_role(self, entity_role: str):
        entity_role_record = await self.driver.query("MERGE (e:EntityRole {text: $entity_role}) RETURN e",
                                                     {"entity_role": entity_role})
        logger.info(f"entity_role: {entity_role_record}")
        return entity_role_record

    async def add_entity_type(self, entity_type: str):
        entity_type_record = await self.driver.query(
            "MERGE (e:EntityType {text: $entity_type}) RETURN e", {"entity_type": entity_type})
        logger.info(f"entity_type: {entity_type_record}")
        return entity_type_record

    async def add_knowledge_category(self, knowledge_category: str):
        knowledge_category_record = await self.driver.query(
            "MERGE (k:KnowledgeCategory {text: $knowledge_category}) RETURN k",
            {"knowledge_category": knowledge_category})
        logger.info(f"knowledge_category: {knowledge_category_record}")
        return knowledge_category_record

    async def add_sentence(self, sentence: str):
        sentence = sentence.strip()

        # Generate signature for lookup in PostgreSQL
        sentence_signature = uuid.uuid5(UUID5_NS, sentence)
        async with (AsyncSessionLocal() as rdb_session):
            async with rdb_session.begin():
                sentence_select_stmt = sql_select(Sentence).where(Sentence.sentence_signature == sentence_signature)
                sentence_select_result = await rdb_session.execute(sentence_select_stmt)
                rdb_record = sentence_select_result.scalars().first()

                # Create record if not found
                if not rdb_record:
                    rdb_record = Sentence(text=sentence, sentence_signature=sentence_signature)
                    rdb_session.add(rdb_record)
                    await rdb_session.commit()

            # Get sentence type if not set
            if (rdb_record.sentence_node_type is None
                    or is_stale_or_over_fetch_count_threshold(
                        rdb_record.sentence_openai_parameters_time_changed,
                        rdb_record.sentence_openai_parameters_fetch_count,
                        hours=24
                    )):
                openai_parameters = await run_in_threadpool(sentence_classifier.classify, sentence)
                async with rdb_session.begin():
                    await rdb_session.refresh(rdb_record)
                    rdb_record.sentence_node_type = SENTENCE_NODE_TYPE[openai_parameters["functional_type"]]
                    rdb_record.sentence_openai_parameters = openai_parameters
                    rdb_record.sentence_openai_parameters_time_changed = utc_now()
                    rdb_record.sentence_openai_parameters_fetch_count += 1
                    await rdb_session.commit()
            logger.info(f"openai_sentence_features: {rdb_record.sentence_openai_parameters}")

            # Merge sentence node
            parameter_spec = sentence_classifier.tool_properties_spec
            sentence_node_type = rdb_record.sentence_node_type
            graph_record = await self.driver.query(
                ("MERGE (s:"
                 + rdb_record.sentence_node_type
                 + " {text: $text, sentence_id: $sentence_id, "
                 + ", ".join([f"{k}: ${k}" for k in parameter_spec.keys()])
                 + "}) RETURN s"),
                {
                    "text": sentence,
                    "sentence_id": rdb_record.sentence_id,
                    **{k: rdb_record.sentence_openai_parameters[k] for k in parameter_spec.keys()},
                })
            logger.info(f"sentence: {graph_record}")
        async_tasks = []
        for handler in self.declarative_sentence_merge_event_handlers:
            async_tasks.append(asyncio.create_task(
                handler.on_merge(sentence_node_type, rdb_record.sentence_id, rdb_record.sentence_openai_parameters)))
        return graph_record, sentence_node_type, rdb_record.sentence_id, async_tasks

    async def add_sentence_subject(self, subject: str):
        subject_record = await self.driver.query("MERGE (n:SentenceSubject {text: $subject}) RETURN n",
                                           {"subject": subject})
        logger.info(f"sentence_subject: {subject_record}")
        return subject_record

    async def add_sentiment_label(self, sentiment: str):
        sentiment_record = await self.driver.query("MERGE (n:SentimentLabel {text: $sentiment}) RETURN n",
                                             {"sentiment": sentiment})
        logger.info(f"sentiment_label: {sentiment_record}")
        return sentiment_record

    async def add_time(self):
        """
        Time nodes are created in set intervals and changed together to match the directionality of real time.
        The idea, here, is that every node created should be linked to its proper chunk of time. This provides a sense
        of temporality.

        :return:
        """
        now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
        current_rounded_time = round_time_now_down_to_nearst_15(now)
        current_time_record = await self.driver.query(
            "MERGE (t:Time {text: $time}) RETURN t", {"time": str(current_rounded_time)})
        logger.info(f"time: {current_time_record}")
        last_rounded_time = current_rounded_time - timedelta(minutes=15)
        time_record_links = await self.driver.query(
            "MATCH (n:Time {text: $now}), (l:Time {text: $last}) "
            "MERGE (n)-[rp:FOLLOWS]->(l) "
            "MERGE (l)-[rf:PRECEDES]->(n) "
            "RETURN rp, rf", {"now": str(current_rounded_time), "last": str(last_rounded_time)})
        logger.debug(f"time links: {time_record_links}")
        return current_rounded_time, current_time_record, time_record_links

    async def add_topic_label(self, topic: str):
        topic_record = await self.driver.query("MERGE (e:TopicLabel {text: $topic}) RETURN e",
                                         {"topic": topic})
        logger.info(f"topic_label: {topic_record}")
        return topic_record

    async def add_verb(self, verb: str):
        verb_record = await self.driver.query("MERGE (v:Verb {text: $verb}) RETURN v",
                                        {"verb": verb})
        logger.info(f"verb: {verb_record}")
        return verb_record

    async def close(self):
        await self.driver.close()

    async def delete_all_data(self):
        await self.driver.delete_all_data()

    async def get_similar_but_disconnected_ideas_by_random_topic(self):
        # Get a random topic that is linked to two ideas that are not the same and not connected
        topic_results = await self.driver.query(
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
            idea_results = await self.driver.query(
                "MATCH (idea1:Idea)-[:EXPRESSES]->(s1:DeclarativeSentence)<-[:CONTAINS]-(t:TopicLabel {text: $topic}) "
                "MATCH (idea2:Idea)-[:EXPRESSES]->(s2:DeclarativeSentence)<-[:CONTAINS]-(t) "
                "WHERE idea1 <> idea2 "
                "AND NOT (idea1)-[:RELATED]-(idea2) "
                "AND idea1.text < idea2.text "  # Sorting ensures unique combo
                "RETURN DISTINCT idea1, idea2", {"topic": topic_results[0]["topic"]["text"]})
            return topic_results, idea_results
        return topic_results, []

    async def get_random_ideas(self, count: int = 2):
        results = await self.driver.query(
            "MATCH (idea:Idea) "
            "WITH idea "
            "ORDER BY rand() "
            f"LIMIT {count} "
            "RETURN idea")
        return results

    async def link_emotion_to_emotion(self, emotion: str, synonymous_emotion: int, link_type: str):
        emotion_link_record = await self.driver.query(
            "MATCH (e1:EmotionLabel {text: $emotion}), (e2:EmotionLabel {text: $synonymous_emotion}) "
            "MERGE (e2)-[r:" + link_type + "]->(e1) "
            "RETURN r", {"emotion": emotion, "synonymous_emotion": synonymous_emotion})
        logger.info(f"emotion/emotion link: {emotion_link_record}")
        return emotion_link_record

    async def link_emotion_to_entity(self, emotion: str, entity: str, sentence_id: int, link_type: str):
        emotion_link_record = await self.driver.query(
            "MATCH (e:EmotionLabel {text: $emotion}), (n:Entity {text: $entity}) "
            # TODO: Implement retention policy on emotion->entity link based on age.
            # The timestamp() means every time words are seen, they are "felt".
            "MERGE (n)-[r:" + link_type + " {sentence_id: $sentence_id, time_evoked: timestamp()}]->(e) "
            "RETURN r", {
                "emotion": emotion, "entity": entity, "sentence_id": sentence_id,
            })
        logger.info(f"emotion link: {emotion_link_record}")
        return emotion_link_record

    async def link_emotion_to_sentence(self, emotion: str, sentence_id: int, sentence_node_type: str):
        emotion_link_record = await self.driver.query(
            "MATCH (e:EmotionLabel {text: $emotion}), (s:" + sentence_node_type + " {sentence_id: $sentence_id}) "
            "MERGE (s)-[r:EVOKED {sentence_id: $sentence_id}]->(e) "
            "RETURN r", {
                "emotion": emotion, "sentence_id": sentence_id,
            })
        logger.info(f"emotion link: {emotion_link_record}")
        return emotion_link_record

    async def link_entity_to_sentence(self, entity: str, sentence_id: int, sentence_node_type: str):
        entity_link_record = await self.driver.query(
            "MATCH (e:Entity {text: $entity}), (s:" + sentence_node_type + " {sentence_id: $sentence_id}) "
            "MERGE (s)-[r:REFERENCES {sentence_id: $sentence_id}]->(e) "
            "RETURN r", {"entity": entity, "sentence_id": sentence_id})
        logger.info(f"entity link: {entity_link_record}")
        return entity_link_record

    async def link_entity_to_sentiment(self, entity: str, sentiment: str, sentence_id: int):
        entity_link_record = await self.driver.query(
            "MATCH (e:Entity {text: $entity}), (s:SentimentLabel {text: $sentiment}) "
            "MERGE (e)-[r:EVOKED {sentence_id: $sentence_id, time_evoked: timestamp()}]->(s) "
            "RETURN r", {"entity": entity, "sentiment": sentiment, "sentence_id": sentence_id})
        logger.info(f"entity/sentiment link: {entity_link_record}")
        return entity_link_record

    async def link_entity_role_to_entity(self, entity: str, entity_role: str, sentence_id: int):
        entity_role_link_record = await self.driver.query(
            "MATCH (e:Entity {text: $entity}), (er:EntityRole {text: $entity_role}) "
            "MERGE (e)-[r:ROLE_OF {sentence_id: $sentence_id}]->(er) "
            "RETURN r", {"entity": entity, "entity_role": entity_role, "sentence_id": sentence_id})
        logger.info(f"entity role link: {entity_role_link_record}")
        return entity_role_link_record

    async def link_entity_type_to_entity(self, entity: str, entity_type: str):
        entity_type_link_record = await self.driver.query(
            "MATCH (e:Entity {text: $entity}), (t:EntityType {text: $entity_type}) "
            "MERGE (e)-[r:TYPE_OF]->(t) "
            "RETURN r", {"entity": entity, "entity_type": entity_type})
        logger.info(f"entity type link: {entity_type_link_record}")
        return entity_type_link_record

    async def link_knowledge_category_to_sentence(self, knowledge_category: str, sentence: str, sentence_node_type: str):
        knowledge_category_link_record = await self.driver.query(
            "MATCH (k:KnowledgeCategory {text: $knowledge_category}), (s:" + sentence_node_type + " {text: $sentence}) "
            "MERGE (s)-[r:CONTEXT_FROM]->(k) "
            "RETURN r", {"knowledge_category": knowledge_category, "sentence": sentence})
        logger.info(f"knowledge category link: {knowledge_category_link_record}")

    async def link_sentiment_to_sentence(self, sentiment: str, sentence: str, sentence_node_type: str):
        sentiment_link_record = await self.driver.query(
            "MATCH (n:SentimentLabel {text: $sentiment}), (s:" + sentence_node_type + " {text: $sentence}) "
            "MERGE (s)-[r:EVOKED]->(n) "
            "RETURN r", {"sentiment": sentiment, "sentence": sentence})
        logger.info(f"sentiment link: {sentiment_link_record}")
        return sentiment_link_record

    async def link_sentence_subject_to_entity(self, sentence_subject: str, entity: str):
        sentence_subject_link_record = await self.driver.query(
            "MATCH (s:SentenceSubject {text: $subject}), (e:Entity {text: $entity}) "
            "MERGE (s)-[r:REFERS_TO]->(e) "
            "RETURN r", {"subject": sentence_subject, "entity": entity})
        logger.info(f"subject/entity link: {sentence_subject_link_record}")
        return sentence_subject_link_record

    async def link_sentence_subject_to_sentence(self, sentence_subject: str, sentence: str, sentence_node_type: str):
        sentence_subject_link_record = await self.driver.query(
            "MATCH (u:SentenceSubject {text: $sentence_subject}), (s:" + sentence_node_type + " {text: $sentence}) "
            "MERGE (s)-[r:IS_ABOUT]->(u) "
            "RETURN r", {"sentence_subject": sentence_subject, "sentence": sentence})
        logger.info(f"subject link: {sentence_subject_link_record}")
        return sentence_subject_link_record

    async def link_topic_label_to_knowledge_category(self, topic_label: str, knowledge_category: str):
        topic_label_link_record = await self.driver.query(
            "MATCH (t:TopicLabel {text: $topic_label}), (k:KnowledgeCategory {text: $knowledge_category}) "
            "MERGE (t)-[r:IS_ABOUT]->(k) "
            "RETURN r", {"topic_label": topic_label, "knowledge_category": knowledge_category})
        logger.info(f"topic_label/knowledge_category link: {topic_label_link_record}")
        return topic_label_link_record

    async def link_topic_label_to_sentence(self, topic_label: str, sentence: str, sentence_node_type: str):
        topic_label_link_record = await self.driver.query(
            "MATCH (t:TopicLabel {text: $topic_label}), (s:" + sentence_node_type + " {text: $sentence}) "
            "MERGE (s)-[r:IS_ABOUT]->(t) "
            "RETURN r", {"topic_label": topic_label, "sentence": sentence})
        logger.info(f"topic_label link: {topic_label_link_record}")
        return topic_label_link_record

    async def link_verb_to_sentence(self, verb: str, sentence: str, sentence_node_type: str):
        verb_link_record = await self.driver.query(
            "MATCH (v:Verb {text: $verb}), (s:" + sentence_node_type + " {text: $sentence}) "
            "MERGE (s)-[r:CONTAINS]->(v) "
            "RETURN r", {"verb": verb, "sentence": sentence})
        logger.info(f"verb link: {verb_link_record}")
        return verb_link_record

    async def merge_ideas(self, idea_to_keep: str, idea_to_merge: str):
        results = await self.driver.query("""
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
        asyncio.run(self.close())


def is_stale_or_over_fetch_count_threshold(dt, fetch_count: int, hours: int, max_fetch_count: int = 25):
    """

    :param dt:  Time changed Datetime object
    :param fetch_count: Number of times fetched from OpenAI
    :param hours: Number of hours to wait before fetching again
    :param max_fetch_count: Maximum number of times fetched from OpenAI since the value diminishes.
    :return:
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return (fetch_count >= max_fetch_count  # Value of fetching diminishes
            or (utc_now() - dt) > timedelta(hours=hours))


def remove_leading_the(text: str):
    return re.sub(r'(?i)^the\s+', '', text)


def round_time_now_down_to_nearst_15(dt):
    minutes = dt.minute
    remainder = minutes % 15  # How many 15-minute increments have passed
    rounded_minutes = minutes - remainder  # Subtract to round down
    return dt.replace(minute=rounded_minutes, second=0, microsecond=0)


def utc_now():
    return datetime.now(timezone.utc)


idea_graph = IdeaGraph(AsyncNeo4jDriver())
signal.signal(signal.SIGINT, idea_graph.signal_handler)


async def main(args):
    await idea_graph.add_default_entity_types()

    if args.load_examples:
        for doc_title, doc_body in doc_examples.all_examples:
            await idea_graph.add_document(doc_title, doc_body)

        # Add sentences to the graph
        #for exp in sentence_examples.all_examples:
        #    await idea_graph.add_sentence(exp)

    await idea_graph.close()


if __name__ == '__main__':
    import argparse

    setup_logging(global_level="INFO")

    parser = argparse.ArgumentParser(description='Build a graph of ideas.')
    parser.add_argument("--load-examples", action="store_true", help='Load example data.',
                        default=False)
    asyncio.run(main(parser.parse_args()))
