import asyncio
import logging
from neo4j import AsyncGraphDatabase
from datetime import datetime

from germ.settings.germ_settings import NEO4J_AUTH, NEO4J_HOST, NEO4J_PORT

logger = logging.getLogger(__name__)


async def _add_chat_message(tx, conversation_id: int, dt_created: datetime):
    query = """
    MERGE (:ChatMessage {conversation_id: $conversation_id, dt_created: $dt_created})
    """
    await tx.run(query, conversation_id=conversation_id, dt_created=dt_created)


async def _add_conversation(tx, conversation_id: int, dt_created: datetime):
    query = """
    MERGE (:Conversation {conversation_id: $conversation_id, dt_created: $dt_created})
    """
    await tx.run(query, conversation_id=conversation_id, dt_created=dt_created)


async def _add_chat_user(tx, user_id: int, dt_created: datetime):
    query = """
    MERGE (:ChatUser {user_id: $user_id, dt_created: $dt_created})
    """
    await tx.run(query, user_id=user_id, dt_created=dt_created)


async def _add_search_queries(tx, conversation_id: int, dt_created: datetime,
                              search_query_embeddings: dict[str, list[float]]):
    query = """
    UNWIND $input_structs AS query_struct
    MERGE (s:SearchQuery {text: query_struct.text})
    WITH query_struct, s
    SET s.embedding = query_struct.embedding
    WITH query_struct, s
    MATCH (m:ChatMessage {conversation_id: query_struct.conversation_id, dt_created: query_struct.dt_created})
    WITH m, s
    MERGE (m)-[:SEEKS]->(s)
    """
    input_structs = [
        {
            'conversation_id': conversation_id,
            'dt_created': dt_created,
            'embedding': embedding,
            'text': text,
        } for text, embedding in search_query_embeddings.items()
    ]
    await tx.run(query, input_structs=input_structs)


async def _add_summary(tx, conversation_id: int, dt_created: datetime,
                       summary_embeddings: dict[str, tuple[int, list[float]]]):
    query = """
    UNWIND $input_structs AS summary_struct
    MERGE (s:Summary {text: summary_struct.text})
    WITH summary_struct, s
    SET s.embedding = summary_struct.embedding
    WITH summary_struct, s
    MATCH (m:ChatMessage {conversation_id: summary_struct.conversation_id, dt_created: summary_struct.dt_created})
    WITH summary_struct, m, s
    MERGE (m)-[:HAS_SUMMARY {position: summary_struct.position}]->(s)
    """
    input_structs = [
        {
            'conversation_id': conversation_id,
            'dt_created': dt_created,
            'embedding': embedding,
            'position': position,
            'text': text,
        } for text, (position, embedding) in summary_embeddings.items()
    ]
    await tx.run(query, input_structs=input_structs)


async def _link_chat_message_received_to_chat_user(tx, conversation_id: int, dt_created: datetime, user_id: int):
    query = """
    MATCH (m:ChatMessage {conversation_id: $conversation_id, dt_created: $dt_created}), (u:ChatUser {user_id: $user_id})
    MERGE (u)-[rel:SENT]->(m)
    """
    await tx.run(query, conversation_id=conversation_id, dt_created=dt_created, user_id=user_id)


async def _link_chat_message_received_to_conversation(tx, conversation_id: int, dt_created: datetime, user_id: int):
    query = """
    MATCH (m:ChatMessage {conversation_id: $conversation_id, dt_created: $dt_created}), (c:Conversation {conversation_id: $conversation_id})
    MERGE (m)-[:PART_OF]->(c)
    """
    await tx.run(query, conversation_id=conversation_id, dt_created=dt_created, user_id=user_id)


async def _link_chat_message_sent_to_chat_message_received(
        tx, received_dt_created: datetime, sent_dt_created: datetime, conversation_id: int):
    query = """
    MATCH (r:ChatMessage {conversation_id: $conversation_id, dt_created: $received_dt_created}), (s:ChatMessage {conversation_id: $conversation_id, dt_created: $sent_dt_created})
    MERGE (s)-[:REACTS_TO]->(r)
    """
    await tx.run(query, received_dt_created=received_dt_created, sent_dt_created=sent_dt_created, conversation_id=conversation_id)


async def _link_chat_user_to_conversation(tx, user_id: int, conversation_id: int):
    query = """
    MATCH (u:ChatUser {user_id: $user_id}), (c:Conversation {conversation_id: $conversation_id})
    MERGE (c)-[:WITH]->(u)
    """
    await tx.run(query, user_id=user_id, conversation_id=conversation_id)


async def _match_search_queries_by_similarity(tx, query_vector: list[float],
                                              k: int = 5, min_similarity: float = 0.8):
    query = """
    CALL db.index.vector.queryNodes('searchQueryVector', $k, $query_vector)
    YIELD node, score
    WHERE score >= $min_similarity
    RETURN score,
           node.embedding   AS embedding,
           node.text        AS text
    """
    results = await tx.run(query, k=k, query_vector=query_vector, min_similarity=min_similarity)
    records = []
    async for result in results:
        records.append({
            "embedding": result["embedding"],
            "score": result["score"],
            "text": result["text"],
        })
    return records


async def _match_search_queries_by_text(tx, texts: list[str]):
    query = """
    MATCH (s:SearchQuery)
    WHERE s.text IN $texts
    RETURN s.embedding  AS embedding,
           s.text       AS text
    """
    result = await tx.run(query, texts=texts)
    records = []
    async for result in result:
        records.append({
            "embedding": result["embedding"],
            "text": result["text"],
        })
    return records


class KnowledgeGraph:
    def __init__(self, driver: AsyncGraphDatabase | None = None):
        self.driver = driver if driver is not None else new_async_driver()

    async def add_chat_message(self, conversation_id: int, dt_created: datetime):
        async with self.driver.session() as session:
            await session.execute_write(_add_chat_message, conversation_id, dt_created)

    async def add_chat_user(self, user_id: int, dt_created: datetime):
        async with self.driver.session() as session:
            await session.execute_write(_add_chat_user, user_id, dt_created)

    async def add_conversation(self, conversation_id: int, dt_created: datetime):
        async with self.driver.session() as session:
            await session.execute_write(_add_conversation, conversation_id, dt_created)

    async def add_search_queries(self, conversation_id: int, dt_created: datetime,
                                 search_query_embeddings: dict[str, list[float]]):
        async with self.driver.session() as session:
            await session.execute_write(_add_search_queries,
                                        conversation_id, dt_created, search_query_embeddings)

    async def add_summary(self, conversation_id: int, dt_created: datetime,
                          summary_embeddings: dict[str, tuple[int, list[float]]]):
        async with self.driver.session() as session:
            await session.execute_write(_add_summary, conversation_id, dt_created, summary_embeddings)

    async def link_chat_message_received_to_chat_user(
            self, conversation_id: int, dt_created: datetime, user_id: int):
        async with self.driver.session() as session:
            await session.execute_write(_link_chat_message_received_to_chat_user,
                                        conversation_id, dt_created, user_id)

    async def link_chat_message_received_to_conversation(
            self, conversation_id: int, dt_created: datetime, user_id: int):
        async with self.driver.session() as session:
            await session.execute_write(_link_chat_message_received_to_conversation,
                                        conversation_id, dt_created, user_id)

    async def link_chat_message_sent_to_chat_message_received(
            self, received_dt_created: datetime, sent_dt_created: datetime, conversation_id: int):
        async with self.driver.session() as session:
            await session.execute_write(_link_chat_message_sent_to_chat_message_received,
                                        received_dt_created, sent_dt_created, conversation_id)

    async def link_chat_user_to_conversation(self, user_id: int, conversation_id: int):
        async with self.driver.session() as session:
            await session.execute_write(_link_chat_user_to_conversation,
                                        user_id=user_id, conversation_id=conversation_id)

    async def match_search_queries_by_similarity(self, query_vector: list[float],
                                                 k: int = 5, min_similarity: float = 0.8):
        async with self.driver.session() as session:
            return await session.execute_read(_match_search_queries_by_similarity,
                                              query_vector, k=k, min_similarity=min_similarity)

    async def match_search_queries_by_text(self, texts: list[str]):
        async with self.driver.session() as session:
            return await session.execute_read(_match_search_queries_by_text, texts=texts)

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
