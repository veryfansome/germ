import json
import re
import signal
import time
import uuid
from datetime import datetime, timedelta
from sqlalchemy.future import select as sql_select
from starlette.concurrency import run_in_threadpool

from bot.lang.utils import (flair_text_feature_extraction, openai_detect_sentence_type,
                            extract_openai_emotion_features, extract_openai_entity_features,
                            openai_text_feature_extraction, split_to_sentences)
from bot.lang.examples import documents as doc_examples, sentences as sentence_examples
from db.neo4j import AsyncNeo4jDriver
from db.models import AsyncSessionLocal, Sentence
from observability.logging import logging, setup_logging

logger = logging.getLogger(__name__)
uuid5_namespace = uuid.UUID("246a5463-afae-4571-a6e0-f319d74147d3")  # Changes sentences signatures

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
    def __init__(self, driver: AsyncNeo4jDriver):
        self.driver = driver

    async def add_document(self, name: str, body: str):
        """
        An article, a poem, or even a chapter of a book.

        :param name:
        :param body:
        :return:
        """
        current_rounded_time, _, _ = await self.add_time()
        document_record = await self.driver.query(
            "MERGE (d:Document {name: $name}) RETURN d", {"name": name})
        logger.info(f"document: {document_record}")
        time_record_link = await self.driver.query(
            "MATCH (t:Time {text: $time}), (d:Document {name: $name}) "
            "MERGE (d)-[r:OCCURRED]->(t) "
            "RETURN r", {"time": str(current_rounded_time), "name": name})
        logger.debug(f"document/time link: {time_record_link}")
        for body_sentence in split_to_sentences(body):
            _, body_sentence_node_type = await self.add_sentence(body_sentence,
                                                                 current_rounded_time=current_rounded_time)
            body_sentence_link_record = await self.driver.query(
                "MATCH (d:Document {name: $document}), (s:" + body_sentence_node_type + " {text: $sentence}) "
                "MERGE (s)-[r:CONTAINS]->(d) "
                "RETURN r", {
                    "document": name, "sentence": body_sentence,
                })
            logger.info(f"sentence link: {body_sentence_link_record}")
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

    async def add_idea(self, idea: str, sentence_id: int, current_rounded_time=None):
        if current_rounded_time is None:
            current_rounded_time, _, _ = await self.add_time()
        idea_record = await self.driver.query(
            "MERGE (i:Idea {text: $idea, sentence_id: $sentence_id}) RETURN i",
            {"idea": idea, "sentence_id": sentence_id})
        logger.info(f"idea: {idea_record}")
        time_record_link = await self.driver.query(
            "MATCH (t:Time {text: $time}), (i:Idea {text: $idea}) "
            "MERGE (i)-[r:OCCURRED {sentence_id: $sentence_id}]->(t) RETURN r",
            {"time": str(current_rounded_time), "idea": idea, "sentence_id": sentence_id})
        logger.debug(f"idea/time link: {time_record_link}")
        return idea_record

    async def add_knowledge_category(self, knowledge_category: str):
        knowledge_category_record = await self.driver.query(
            "MERGE (k:KnowledgeCategory {text: $knowledge_category}) RETURN k",
            {"knowledge_category": knowledge_category})
        logger.info(f"knowledge_category: {knowledge_category_record}")
        return knowledge_category_record

    async def add_sentence(self, sentence: str,
                           current_rounded_time=None, flair_features=None, openai_features=None,
                           sentence_node_type=None):
        sentence = sentence.strip()
        if current_rounded_time is None:
            current_rounded_time, _, _ = await self.add_time()

        # Generate signature for lookup in PostgreSQL
        sentence_signature = uuid.uuid5(uuid5_namespace, sentence)
        async with AsyncSessionLocal() as rdb_session:
            async with rdb_session.begin():
                sentence_select_stmt = sql_select(Sentence).where(Sentence.sentence_signature == sentence_signature)
                sentence_select_result = await rdb_session.execute(sentence_select_stmt)
                sentence_rdb_record = sentence_select_result.scalars().first()

                # Create record if not found
                if not sentence_rdb_record:
                    sentence_rdb_record = Sentence(text=sentence, sentence_signature=sentence_signature)
                    rdb_session.add(sentence_rdb_record)
                    await rdb_session.commit()

            # Get sentence type if not set
            if sentence_rdb_record.sentence_node_type is None:
                openai_sentence_type = (
                    {"sentence_type": sentence_node_type}
                    if sentence_node_type is not None else json.loads(await run_in_threadpool(
                        openai_detect_sentence_type, sentence, model="gpt-4o-mini")))
                async with rdb_session.begin():
                    await rdb_session.refresh(sentence_rdb_record)
                    sentence_rdb_record.sentence_node_type = SENTENCE_NODE_TYPE[openai_sentence_type["sentence_type"]]
                    await rdb_session.commit()

            # Create sentence node and link to current time node
            sentence_node_type = sentence_rdb_record.sentence_node_type
            sentence_graph_record = await self.driver.query(
                "MERGE (s:" + sentence_node_type + " {text: $text, sentence_id: $sentence_id}) RETURN s",
                {"text": sentence, "sentence_id": sentence_rdb_record.sentence_id})
            logger.info(f"sentence: {sentence_graph_record}")
            time_record_link = await self.driver.query(
                "MATCH (t:Time {text: $time}), (s:" + sentence_node_type + " {text: $sentence}) "
                "MERGE (s)-[r:OCCURRED {sentence_id: $sentence_id}]->(t) "
                "RETURN r", {
                    "time": str(current_rounded_time), "sentence": sentence,
                    "sentence_id": sentence_rdb_record.sentence_id})
            logger.debug(f"sentence/time link: {time_record_link}")

            # Create a linked idea because every sentence expresses at least one idea.
            await self.add_idea(sentence, sentence_rdb_record.sentence_id, current_rounded_time=current_rounded_time)
            idea_link_record = await self.driver.query(
                "MATCH (i:Idea {text: $idea}), (s:" + sentence_node_type + " {text: $sentence}) "
                "MERGE (s)-[r:EXPRESSES {sentence_id: $sentence_id}]->(i) "
                "RETURN r", {
                    "idea": sentence, "sentence": sentence,
                    "sentence_id": sentence_rdb_record.sentence_id})
            logger.info(f"idea link: {idea_link_record}")

            entity_set = set()
            entity_token_set = set()
            knowledge_category_set = set()

            # Get emotion features
            if sentence_rdb_record.sentence_openai_emotion_features is None:
                openai_emotion_features = json.loads(await run_in_threadpool(
                    extract_openai_emotion_features, sentence, model="gpt-4o-mini"))
                async with rdb_session.begin():
                    await rdb_session.refresh(sentence_rdb_record)
                    sentence_rdb_record.sentence_openai_emotion_features = openai_emotion_features
                    await rdb_session.commit()

            # Emotion
            openai_emotions_features = sentence_rdb_record.sentence_openai_emotion_features
            for emotion in openai_emotions_features["emotions"]:
                logger.info(f"emotion: {emotion}")
                if emotion["emotion"] == "neutral":
                    continue  # Skip neutral to focus on the not neutral

                await self.add_emotion(emotion["emotion"])
                await self.link_emotion_to_sentence(
                    emotion["emotion"], sentence_rdb_record.sentence_id, sentence_node_type,
                    intensity=emotion["intensity"], nuance=emotion["nuance"])

                # Add emotion source to entities
                emotion_source = remove_leading_the(emotion["emotion_source"])
                entity_set.add(emotion_source)
                for token in emotion_source.split():
                    entity_token_set.add(token)
                await self.add_entity(emotion_source)
                await self.link_emotion_to_entity(
                    emotion["emotion"], emotion_source, sentence_rdb_record.sentence_id, "FELT",
                    intensity=emotion["intensity"], nuance=emotion["nuance"])
                await self.link_entity_to_sentence(
                    emotion_source, sentence_rdb_record.sentence_id, sentence_node_type)
                await self.add_entity_type(emotion["emotion_source_entity_type"])
                await self.link_entity_type_to_entity(emotion_source, emotion["emotion_source_entity_type"])

                # Add emotion target to entities
                emotion_target = remove_leading_the(emotion["emotion_target"])
                entity_set.add(emotion_target)
                for token in emotion_target.split():
                    entity_token_set.add(token)
                await self.add_entity(emotion_target)
                await self.link_emotion_to_entity(
                    emotion["emotion"], emotion_target, sentence_rdb_record.sentence_id, "EVOKED",
                    intensity=emotion["intensity"], nuance=emotion["nuance"])
                await self.link_entity_to_sentence(
                    emotion_target, sentence_rdb_record.sentence_id, sentence_node_type)
                await self.add_entity_type(emotion["emotion_target_entity_type"])
                await self.link_entity_type_to_entity(emotion_target, emotion["emotion_target_entity_type"])

                # Add and link similar emotions
                for synonymous_emotion in emotion["synonymous_emotions"]:
                    await self.add_emotion(synonymous_emotion)
                    await self.link_emotion_to_emotion(emotion["emotion"], synonymous_emotion, "SYNONYMOUS")

                # Add and link opposite emotions
                for opposite_emotion in emotion["opposite_emotions"]:
                    await self.add_emotion(opposite_emotion)
                    await self.link_emotion_to_emotion(emotion["emotion"], opposite_emotion, "OPPOSITE")

            # Get flair text features
            if sentence_rdb_record.sentence_flair_text_features is None:
                flair_features = (
                    flair_features if flair_features is not None else await run_in_threadpool(
                        flair_text_feature_extraction, sentence))
                async with rdb_session.begin():
                    await rdb_session.refresh(sentence_rdb_record)
                    sentence_rdb_record.sentence_flair_text_features = flair_features
                    await rdb_session.commit()

            # Get OpenAI entity features
            if sentence_rdb_record.sentence_openai_entity_features is None:
                openai_entity_features = json.loads(await run_in_threadpool(
                    extract_openai_entity_features, sentence, model="gpt-4o-mini"))
                async with rdb_session.begin():
                    await rdb_session.refresh(sentence_rdb_record)
                    sentence_rdb_record.sentence_openai_entity_features = openai_entity_features
                    await rdb_session.commit()

            # Entities
            for entity in sentence_rdb_record.sentence_openai_entity_features["entities"]:
                logger.info(f"entity: {entity}")

                # Add entity and link to sentence.
                entity_name = remove_leading_the(entity["entity"])
                entity_set.add(entity_name)
                for token in entity_name.split():
                    entity_token_set.add(token)
                await self.add_entity(entity_name)
                await self.link_entity_to_sentence(entity_name, sentence_rdb_record.sentence_id, sentence_node_type)

                # Add entity type and link entity to is type
                await self.add_entity_type(entity["entity_type"])
                await self.link_entity_type_to_entity(entity_name, entity["entity_type"])

                if "semantic_role" in entity:
                    await self.add_entity_role(entity["semantic_role"])
                    await self.link_entity_role_to_entity(entity_name, entity["semantic_role"],
                                                          sentence_rdb_record.sentence_id)

                # Link entity to a sentiment based on context.
                if "sentiment" in entity and entity["sentiment"] != "neutral":
                    await self.add_sentiment_label(entity["sentiment"])
                    await self.link_entity_to_sentiment(entity_name, entity["sentiment"],
                                                        sentence_rdb_record.sentence_id)

            for entity in sentence_rdb_record.sentence_flair_text_features["ner"]:
                entity_set.add(entity)
                for token in entity.split():
                    entity_token_set.add(token)
                await self.add_entity(entity)
                await self.link_entity_to_sentence(entity, sentence_rdb_record.sentence_id, sentence_node_type)
            for proper_noun in sentence_rdb_record.sentence_flair_text_features["proper_nouns"]:
                if proper_noun not in entity_set and proper_noun not in entity_token_set:
                    entity_set.add(proper_noun)
                    for token in proper_noun.split():
                        entity_token_set.add(token)
                    await self.add_entity(proper_noun)
                    await self.link_entity_to_sentence(proper_noun, sentence_rdb_record.sentence_id, sentence_node_type)

            if sentence_rdb_record.sentence_openai_text_features is None:
                openai_text_features = (
                    openai_features if openai_features is not None
                    else json.loads(openai_text_feature_extraction(sentence)))
                async with rdb_session.begin():
                    await rdb_session.refresh(sentence_rdb_record)
                    sentence_rdb_record.sentence_openai_text_features = openai_text_features
                    await rdb_session.commit()

            # Knowledge category
            for knowledge_category in sentence_rdb_record.sentence_openai_text_features["knowledge"]:
                await self.add_knowledge_category(knowledge_category)
                await self.link_knowledge_category_to_sentence(knowledge_category, sentence, sentence_node_type)

            # Sentiment
            sentiment = sentence_rdb_record.sentence_openai_text_features["sentiment"]
            if sentiment != "neutral":  # Skip neutral to focus on the not neutral
                await self.add_sentiment_label(sentiment)
                await self.link_sentiment_to_sentence(sentiment, sentence, sentence_node_type)

            # Sentence subject
            for subject in sentence_rdb_record.sentence_openai_text_features["subjects"]:
                await self.add_sentence_subject(subject)
                await self.link_sentence_subject_to_sentence(subject, sentence, sentence_node_type)
                if subject in entity_set:
                    await self.link_sentence_subject_to_entity(subject, subject)

            # Topic
            for topic in sentence_rdb_record.sentence_openai_text_features["topics"]:
                await self.add_topic_label(topic)
                await self.link_topic_label_to_sentence(topic, sentence, sentence_node_type)
                if topic in knowledge_category_set:
                    await self.link_topic_label_to_knowledge_category(topic, topic)

            # TODO: Convert this to use OpenAI for get rid of it.
            # Verb
            for verb in sentence_rdb_record.sentence_flair_text_features["verbs"]:
                verb = verb.lower()
                # Filter out states-of-being verbs because these should be modeled as connections between entities.
                if verb in ["be", "become"]:
                    logger.info(f"future verb: {verb}")
                elif verb in ["will"]:
                    logger.info(f"future modal verb: {verb}")
                elif verb in ["became", "been", "was", "were"]:
                    logger.info(f"past verb: {verb}")
                elif verb in ["could", "would"]:
                    logger.info(f"past modal verb: {verb}")
                elif verb in ["had"]:
                    logger.info(f"past possessive verb: {verb}")
                elif verb in ["am", "are", "is", "being"]:
                    logger.info(f"present verb: {verb}")
                elif verb in ["can"]:
                    logger.info(f"present modal verb: {verb}")
                elif verb in ["has", "have"]:
                    logger.info(f"present possessive verb: {verb}")
                else:
                    await self.add_verb(verb)
                    await self.link_verb_to_sentence(verb, sentence, sentence_node_type)

        return sentence_graph_record, sentence_node_type

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
        now = datetime.now().replace(second=0, microsecond=0)
        current_rounded_time = round_time_now_down_to_nearst_15(now)
        current_time_record = await self.driver.query(
            "MERGE (t:Time {text: $time, epoch: $epoch}) RETURN t",
            {"time": str(current_rounded_time), "epoch": int(time.time())})
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

    def close(self):
        self.driver.close()

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

    async def link_emotion_to_entity(self, emotion: str, entity: str, sentence_id: int, link_type: str,
                               intensity: str = "none", nuance: str = "none"):
        emotion_link_record = await self.driver.query(
            "MATCH (e:EmotionLabel {text: $emotion}), (n:Entity {text: $entity}) "
            # TODO: Implement retention policy on emotion->entity link based on age.
            # The timestamp() means every time words are seen, they are "felt".
            "MERGE (n)-[r:" + link_type + " {sentence_id: $sentence_id, intensity: $intensity, nuance: $nuance, time_evoked: timestamp()}]->(e) "
            "RETURN r", {
                "emotion": emotion, "entity": entity,
                "intensity": intensity, "nuance": nuance,
                "sentence_id": sentence_id,
            })
        logger.info(f"emotion link: {emotion_link_record}")
        return emotion_link_record

    async def link_emotion_to_sentence(self, emotion: str, sentence_id: int, sentence_node_type: str,
                                 intensity: str = "none", nuance: str = "none"):
        emotion_link_record = await self.driver.query(
            "MATCH (e:EmotionLabel {text: $emotion}), (s:" + sentence_node_type + " {sentence_id: $sentence_id}) "
            "MERGE (s)-[r:EVOKED {sentence_id: $sentence_id, intensity: $intensity, nuance: $nuance}]->(e) "
            "RETURN r", {
                "emotion": emotion, "sentence_id": sentence_id,
                "intensity": intensity, "nuance": nuance,
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
            "MERGE (v)-[r:CONTAINS]->(s) "
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
        self.close()


def get_idea_graph(name: str):
    if name not in INSTANCE_MANIFEST:
        new_instance = IdeaGraph(AsyncNeo4jDriver())
        signal.signal(signal.SIGINT, new_instance.signal_handler)
        INSTANCE_MANIFEST[name] = new_instance
    return INSTANCE_MANIFEST[name]


def remove_leading_the(text: str):
    return re.sub(r'(?i)^the\s+', '', text)


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
        #for exp in sentence_examples.all_examples:
        #    idea_graph.add_sentence(exp)

    idea_graph.close()
