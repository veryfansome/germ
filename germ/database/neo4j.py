import asyncio
import logging
from neo4j import AsyncGraphDatabase
from datetime import datetime

from germ.settings.germ_settings import NEO4J_AUTH, NEO4J_HOST, NEO4J_PORT

logger = logging.getLogger(__name__)


async def _add_chat_message(tx, conversation_id: int, dt_created: datetime):
    query = """
    MERGE (message:ChatMessage {conversation_id: $conversation_id, dt_created: $dt_created})
    RETURN message
    """
    return await tx.run(query, conversation_id=conversation_id, dt_created=dt_created)


async def _add_conversation(tx, conversation_id: int, dt_created: datetime):
    query = """
    MERGE (conversation:Conversation {conversation_id: $conversation_id, dt_created: $dt_created})
    RETURN conversation
    """
    return await tx.run(query, conversation_id=conversation_id, dt_created=dt_created)


async def _add_chat_user(tx, user_id: int, dt_created: datetime):
    query = """
    MERGE (user:ChatUser {user_id: $user_id, dt_created: $dt_created})
    RETURN user
    """
    return await tx.run(query, user_id=user_id, dt_created=dt_created)


async def _link_chat_message_received_to_chat_user(tx, conversation_id: int, dt_created: datetime, user_id: int):
    query = """
    MATCH (message:ChatMessage {conversation_id: $conversation_id, dt_created: $dt_created}), (user:ChatUser {user_id: $user_id})
    MERGE (user)-[rel:SENT]->(message)
    RETURN rel
    """
    return await tx.run(query, conversation_id=conversation_id, dt_created=dt_created, user_id=user_id)


async def _link_chat_message_sent_to_chat_message_received(
        tx, received_dt_created: datetime, sent_dt_created: datetime, conversation_id: int):
    query = """
    MATCH (received:ChatMessage {conversation_id: $conversation_id, dt_created: $received_dt_created}), (sent:ChatMessage {conversation_id: $conversation_id, dt_created: $sent_dt_created})
    MERGE (sent)-[rel:REACTS_TO]->(received)
    RETURN rel
    """
    return await tx.run(query, received_dt_created=received_dt_created, sent_dt_created=sent_dt_created, conversation_id=conversation_id)


async def _link_chat_user_to_conversation(tx, user_id: int, conversation_id: int):
    query = """
    MATCH (user:ChatUser {user_id: $user_id}), (conversation:Conversation {conversation_id: $conversation_id})
    MERGE (conversation)-[rel:WITH]->(user)
    RETURN rel
    """
    return await tx.run(query, user_id=user_id, conversation_id=conversation_id)


async def _match_all_topic_definitions(tx):
    query = """
    MATCH (n)-[:OF_TOPIC]->(t)
    MATCH (d)-[r:DEFINES]->(t)
    RETURN DISTINCT(d),r,t
    """
    results = []
    async for record in await tx.run(query):
        results.append(record)
    return results


async def _match_synset_definition(tx, lemma: str, pos: str):
    query = """
    MATCH (syndef)-[rel:DEFINES]-(syn {pos: $pos})
    WHERE $lemma IN syn.lemmas
    RETURN syndef,rel,syn
    """
    results = []
    async for record in await tx.run(query, lemma=lemma, pos=pos):
        results.append(record)
    return results


class KnowledgeGraph:
    def __init__(self, driver: AsyncGraphDatabase | None = None):
        self.driver = driver if driver is not None else new_async_driver()

    async def add_chat_message(self, conversation_id: int, dt_created: datetime):
        async with self.driver.session() as session:
            results = await session.execute_write(
                _add_chat_message, conversation_id=conversation_id, dt_created=dt_created)
            if results:
                logger.info(f"MERGE (message:ChatMessage {{conversation_id: {conversation_id}, dt_created: {dt_created}}})")
            return results

    async def add_chat_user(self, user_id: int, dt_created: datetime):
        async with self.driver.session() as session:
            results = await session.execute_write(
                _add_chat_user, user_id=user_id, dt_created=dt_created)
            if results:
                logger.info(f"MERGE (user:ChatUser {{user_id: {user_id}, dt_created: {dt_created}}})")
            return results

    async def add_conversation(self, conversation_id: int, dt_created: datetime):
        async with self.driver.session() as session:
            results = await session.execute_write(
                _add_conversation, conversation_id=conversation_id, dt_created=dt_created)
            if results:
                logger.info(f"MERGE (conversation:Conversation {{conversation_id: {conversation_id}, dt_created: {dt_created}}})")
            return results

    async def link_chat_message_received_to_chat_user(self, conversation_id: int, dt_created: datetime, user_id: int):
        async with self.driver.session() as session:
            results = await session.execute_write(
                _link_chat_message_received_to_chat_user,
                conversation_id=conversation_id, dt_created=dt_created, user_id=user_id)
            if results:
                logger.info("MERGE "
                            f"(user:ChatUser {{user_id: {user_id}}})-[rel:SENT]->"
                            f"(message:ChatMessage {{conversation_id: {conversation_id}, dt_created: {dt_created}}})")
            return results

    async def link_chat_message_sent_to_chat_message_received(
            self, received_dt_created: datetime, sent_dt_created: datetime, conversation_id: int):
        async with self.driver.session() as session:
            results = await session.execute_write(
                _link_chat_message_sent_to_chat_message_received,
                received_dt_created=received_dt_created, sent_dt_created=sent_dt_created,
                conversation_id=conversation_id)
            if results:
                logger.info("MERGE "
                            f"(sent:ChatMessage {{conversation_id: {conversation_id}, dt_created: {sent_dt_created}}})-[rel:REACTS_TO]->"
                            f"(received:ChatMessage {{conversation_id: {conversation_id}, dt_created: {received_dt_created}}})")
            return results

    async def link_chat_user_to_conversation(self, user_id: int, conversation_id: int):
        async with self.driver.session() as session:
            results = await session.execute_write(
                _link_chat_user_to_conversation,
                user_id=user_id, conversation_id=conversation_id)
            if results:
                logger.info("MERGE "
                            f"(conversation:Conversation {{conversation_id: {conversation_id}}})-[rel:WITH]->"
                            f"(user:ChatUser {{user_id: {user_id}}})")
            return results

    async def match_all_topic_definitions(self):
        async with self.driver.session() as session:
            return await session.execute_read(_match_all_topic_definitions)

    async def match_synset_definition(self, lemma: str, pos: str):
        async with self.driver.session() as session:
            return await session.execute_read(_match_synset_definition, lemma=lemma, pos=pos)

    async def shutdown(self):
        if self.driver:
            await self.driver.close()
            self.driver = None
        else:
            logger.warning("shutdown called on an already closed driver")


def new_async_driver():
    auth_parts = NEO4J_AUTH.split("/")
    return AsyncGraphDatabase.driver(
        f"bolt://{NEO4J_HOST}:{NEO4J_PORT}", auth=(auth_parts[0], auth_parts[1]))


if __name__ == "__main__":
    from germ.observability.logging import setup_logging
    setup_logging()

    async def main():
        knowledge_graph = KnowledgeGraph()
        await knowledge_graph.shutdown()

    asyncio.run(main())
