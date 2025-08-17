import logging
from neo4j import AsyncGraphDatabase
from datetime import datetime
from traceback import format_exc

from germ.settings.germ_settings import NEO4J_AUTH, NEO4J_HOST, NEO4J_PORT

logger = logging.getLogger(__name__)


async def _add_chat_message(tx, conversation_id: int, dt_created: datetime):
    cypher = """
    MERGE (:ChatMessage {conversation_id: $conversation_id, dt_created: $dt_created})
    """
    await tx.run(cypher, conversation_id=conversation_id, dt_created=dt_created)


async def _add_conversation(tx, conversation_id: int, dt_created: datetime):
    cypher = """
    MERGE (:Conversation {conversation_id: $conversation_id, dt_created: $dt_created})
    """
    await tx.run(cypher, conversation_id=conversation_id, dt_created=dt_created)


async def _add_chat_user(tx, user_id: int, dt_created: datetime):
    cypher = """
    MERGE (:ChatUser {user_id: $user_id, dt_created: $dt_created})
    """
    await tx.run(cypher, user_id=user_id, dt_created=dt_created)


async def _add_keyword_phrases(
        tx, conversation_id: int, dt_created: datetime,
        keyword_phrase_embeddings: dict[str, list[float]]
):
    cypher = """
    UNWIND $input_structs AS input_struct
    MERGE (kw:KeywordPhrase {text: input_struct.text})
    WITH input_struct, kw
    SET kw.embedding = input_struct.embedding
    WITH input_struct, kw
    MATCH (m:ChatMessage {conversation_id: input_struct.conversation_id, dt_created: input_struct.dt_created})
    WITH m, kw
    MERGE (m)-[:SEEKS]->(kw)
    """
    input_structs = [
        {
            'conversation_id': conversation_id,
            'dt_created': dt_created,
            'embedding': embedding,
            'text': text,
        } for text, embedding in keyword_phrase_embeddings.items()
    ]
    await tx.run(cypher, input_structs=input_structs)


async def _add_summary(tx, conversation_id: int, dt_created: datetime,
                       summary_embeddings: dict[str, tuple[int, list[float]]]):
    cypher = """
    UNWIND $input_structs AS input_struct
    MERGE (s:Summary {text: input_struct.text})
    WITH input_struct, s
    SET s.embedding = input_struct.embedding
    WITH input_struct, s
    MATCH (m:ChatMessage {conversation_id: input_struct.conversation_id, dt_created: input_struct.dt_created})
    WITH input_struct, m, s
    MERGE (m)-[:HAS_SUMMARY {position: input_struct.position}]->(s)
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
    await tx.run(cypher, input_structs=input_structs)


async def _add_user_message_intent(tx, conversation_id: int, dt_created: datetime,
                                   labels: list[tuple[str, str]]):
    cypher = """
    MATCH (m:ChatMessage {conversation_id: $conversation_id, dt_created: $dt_created})
    UNWIND $input_structs AS input_struct
    MERGE (i:Intent {name: input_struct.intent})
    WITH input_struct, m, i
    MERGE (c:IntentCategory {name: input_struct.category})
    WITH m, i, c
    MERGE (m)-[:HAS_INTENT]->(i)
    WITH i, c
    MERGE (i)-[:CATEGORIZED_AS]->(c)
    """
    input_structs = [
        {
            'category': category,
            'intent': intent,
        } for category, intent in labels
    ]
    await tx.run(cypher, conversation_id=conversation_id, dt_created=dt_created, input_structs=input_structs)


async def _link_chat_message_received_to_chat_user(tx, conversation_id: int, dt_created: datetime, user_id: int):
    cypher = """
    MATCH (m:ChatMessage {conversation_id: $conversation_id, dt_created: $dt_created}), (u:ChatUser {user_id: $user_id})
    MERGE (u)-[rel:SENT]->(m)
    """
    await tx.run(cypher, conversation_id=conversation_id, dt_created=dt_created, user_id=user_id)


async def _link_chat_message_received_to_conversation(tx, conversation_id: int, dt_created: datetime, user_id: int):
    cypher = """
    MATCH (m:ChatMessage {conversation_id: $conversation_id, dt_created: $dt_created}), (c:Conversation {conversation_id: $conversation_id})
    MERGE (m)-[:PART_OF]->(c)
    """
    await tx.run(cypher, conversation_id=conversation_id, dt_created=dt_created, user_id=user_id)


async def _link_chat_message_sent_to_chat_message_received(
        tx, received_dt_created: datetime, sent_dt_created: datetime, conversation_id: int):
    cypher = """
    MATCH (r:ChatMessage {conversation_id: $conversation_id, dt_created: $received_dt_created}), (s:ChatMessage {conversation_id: $conversation_id, dt_created: $sent_dt_created})
    MERGE (s)-[:REACTS_TO]->(r)
    """
    await tx.run(cypher, received_dt_created=received_dt_created, sent_dt_created=sent_dt_created, conversation_id=conversation_id)


async def _link_chat_message_to_summary(tx, conversation_id: int, dt_created: datetime, position: int, summary_text: str):
    cypher = """
    MATCH (m: ChatMessage {conversation_id: $conversation_id, dt_created: $dt_created}), (s:Summary {text: $summary_text})
    MERGE (m)-[:HAS_SUMMARY {position: $position}]->(s)
    """
    await tx.run(cypher, conversation_id=conversation_id, dt_created=dt_created, position=position, summary_text=summary_text)


async def _link_chat_user_to_conversation(tx, user_id: int, conversation_id: int):
    cypher = """
    MATCH (u:ChatUser {user_id: $user_id}), (c:Conversation {conversation_id: $conversation_id})
    MERGE (c)-[:WITH]->(u)
    """
    await tx.run(cypher, user_id=user_id, conversation_id=conversation_id)


async def _match_bot_message_summaries_by_similarity_to_query_vector(
        tx, query_vector: list[float], k: int = 5, min_similarity: float = 0.8
):
    cypher = """
    MATCH (m:ChatMessage), (s:Summary)
    WHERE NOT (:ChatUser)-[:SENT]->(m) AND (m)-[:HAS_SUMMARY]->(s) AND s.embedding IS NOT NULL
    WITH m, s, vector.similarity.cosine(s.embedding, $query_vector) AS score
    WHERE score >= $min_similarity
    ORDER BY score DESC LIMIT $k
    RETURN score,
           m.conversation_id    AS conversation_id,
           m.dt_created         AS dt_created,
           s.embedding          AS embedding,
           s.text               AS text
    """
    results = await tx.run(cypher, query_vector=query_vector, k=k, min_similarity=min_similarity)
    records = []
    async for result in results:
        records.append({
            "conversation_id": result["conversation_id"],
            "dt_created": result["dt_created"],
            "embedding": result["embedding"],
            "score": result["score"],
            "text": result["text"],
        })
    return records


async def _match_keyword_phrases_by_similarity(tx, query_vector: list[float],
                                               k: int = 5, min_similarity: float = 0.8):
    cypher = """
    CALL db.index.vector.queryNodes('searchQueryVector', $k, $query_vector)
    YIELD node, score
    WHERE score >= $min_similarity
    RETURN score,
           node.embedding   AS embedding,
           node.text        AS text
    """
    results = await tx.run(cypher, k=k, query_vector=query_vector, min_similarity=min_similarity)
    records = []
    async for result in results:
        records.append({
            "embedding": result["embedding"],
            "score": result["score"],
            "text": result["text"],
        })
    return records


async def _match_keyword_phrases_by_similarity_to_query_vector(
        tx, query_vector: list[float], similar_message_structs: list[dict[str, float | datetime | int]],
        alpha: float = 0.7, k: int = 5,
):
    cypher = """
    UNWIND $input_structs AS input_struct
    MATCH (m:ChatMessage {conversation_id: input_struct.conversation_id})
    WHERE datetime(m.dt_created).epochMillis = datetime(input_struct.dt_created).epochMillis
    WITH collect(DISTINCT m) AS messages, input_struct
    MATCH (kw:KeywordPhrase)
    WHERE all(m IN messages WHERE (m)-[:SEEKS]->(kw) AND kw.embedding IS NOT NULL)
    WITH DISTINCT kw AS kw,
          coalesce(input_struct.recalled_message_score, 0.0)                                AS recalled_message_score,
          coalesce(vector.similarity.cosine(kw.embedding, $query_vector), 0.0)              AS keyword_score
     WITH kw,
          ((recalled_message_score + 1.0) / 2.0)                                            AS norm_recalled_message_score,
          ((keyword_score + 1.0) / 2.0)                                                     AS norm_keyword_score
     WITH kw,
          (($alpha * norm_recalled_message_score) + ((1 - $alpha) * norm_keyword_score))    AS combined_score
     WITH kw, max(combined_score)                                                           AS score
     ORDER BY score DESC LIMIT $k
     RETURN score,
            kw.embedding    AS embedding,
            kw.text         AS text
    """
    input_structs = [
        {
            "conversation_id": m["conversation_id"],
            "dt_created": m["dt_created"],
            "recalled_message_score": m["score"],
        } for m in similar_message_structs
    ]
    results = await tx.run(cypher, query_vector=query_vector, input_structs=input_structs,
                           alpha=alpha, k=k)
    records = []
    async for result in results:
        records.append({
            "score": result["score"],
            "embedding": result["embedding"],
            "text": result["text"],
        })
    return records


async def _match_keyword_phrases_by_text(tx, texts: list[str]):
    cypher = """
    MATCH (kw:KeywordPhrase)
    WHERE kw.text IN $texts
    RETURN kw.embedding AS embedding,
           kw.text      AS text
    """
    result = await tx.run(cypher, texts=texts)
    records = []
    async for result in result:
        records.append({
            "embedding": result["embedding"],
            "text": result["text"],
        })
    return records


async def _match_user_message_summaries_by_similarity_to_query_vector(
        tx, user_id: int, query_vector: list[float], k: int = 5, min_similarity: float = 0.8
):
    cypher = """
    MATCH (m:ChatMessage), (s:Summary)
    WHERE (:ChatUser {user_id: $user_id})-[:SENT]->(m) AND (m)-[:HAS_SUMMARY]->(s) AND s.embedding IS NOT NULL
    WITH m, s, vector.similarity.cosine(s.embedding, $query_vector) AS score
    WHERE score >= $min_similarity
    ORDER BY score DESC LIMIT $k
    RETURN score,
           m.conversation_id    AS conversation_id,
           m.dt_created         AS dt_created,
           s.embedding          AS embedding,
           s.text               AS text
    """
    results = await tx.run(cypher, user_id=user_id, query_vector=query_vector, k=k, min_similarity=min_similarity)
    records = []
    async for result in results:
        records.append({
            "conversation_id": result["conversation_id"],
            "dt_created": result["dt_created"],
            "embedding": result["embedding"],
            "score": result["score"],
            "text": result["text"],
        })
    return records


class KnowledgeGraph:
    def __init__(self, driver: AsyncGraphDatabase | None = None):
        self.driver = driver if driver is not None else new_async_driver()

    async def add_chat_message(self, conversation_id: int, dt_created: datetime):
        try:
            async with self.driver.session() as session:
                await session.execute_write(_add_chat_message, conversation_id, dt_created)
        except Exception:
            logging.error(f"Failed to add chat message from {conversation_id} at {dt_created}: {format_exc()}")

    async def add_chat_user(self, user_id: int, dt_created: datetime):
        try:
            async with self.driver.session() as session:
                await session.execute_write(_add_chat_user, user_id, dt_created)
        except Exception:
            logging.error(f"Failed to add chat user {user_id}: {format_exc()}")

    async def add_conversation(self, conversation_id: int, dt_created: datetime):
        try:
            async with self.driver.session() as session:
                await session.execute_write(_add_conversation, conversation_id, dt_created)
        except Exception:
            logging.error(f"Failed to add conversation {conversation_id}: {format_exc()}")

    async def add_keyword_phrases(
            self, conversation_id: int, dt_created: datetime,
            keyword_phrase_embeddings: dict[str, list[float]]
    ):
        try:
            async with self.driver.session() as session:
                await session.execute_write(
                    _add_keyword_phrases, conversation_id, dt_created, keyword_phrase_embeddings
                )
        except Exception:
            logging.error(f"Failed to add search keywords related to chat message from conversation {conversation_id}, "
                          f"received at {dt_created}, {keyword_phrase_embeddings.keys()}: {format_exc()}")

    async def add_summary(
            self, conversation_id: int, dt_created: datetime,
            summary_embeddings: dict[str, tuple[int, list[float]]]
    ):
        try:
            async with self.driver.session() as session:
                await session.execute_write(_add_summary, conversation_id, dt_created, summary_embeddings)
        except Exception:
            logging.error(f"Failed to add summaries of chat message from conversation {conversation_id}, "
                          f"received at {dt_created}, {summary_embeddings.keys()}: {format_exc()}")

    async def add_user_message_intent(
            self, conversation_id: int, dt_created: datetime,
            labels: list[tuple[str, str]]
    ):
        try:
            async with self.driver.session() as session:
                await session.execute_write(_add_user_message_intent, conversation_id, dt_created, labels)
        except Exception:
            logging.error(f"Failed to add intent of chat message from conversation {conversation_id}, "
                          f"received at {dt_created}, {labels}: {format_exc()}")


    async def link_chat_message_received_to_chat_user(
            self, conversation_id: int, dt_created: datetime, user_id: int
    ):
        try:
            async with self.driver.session() as session:
                await session.execute_write(_link_chat_message_received_to_chat_user,
                                            conversation_id, dt_created, user_id)
        except Exception:
            logger.error(f"Failed to link chat message from conversation {conversation_id}, "
                         f"received at {dt_created} to chat user {user_id}: {format_exc()}")

    async def link_chat_message_received_to_conversation(
            self, conversation_id: int, dt_created: datetime, user_id: int
    ):
        try:
            async with self.driver.session() as session:
                await session.execute_write(_link_chat_message_received_to_conversation,
                                            conversation_id, dt_created, user_id)
        except Exception:
            logger.error(f"Failed to link chat message from conversation {conversation_id}, "
                         f"received at {dt_created}: {format_exc()}")

    async def link_chat_message_sent_to_chat_message_received(
            self, received_dt_created: datetime, sent_dt_created: datetime, conversation_id: int
    ):
        try:
            async with self.driver.session() as session:
                await session.execute_write(_link_chat_message_sent_to_chat_message_received,
                                            received_dt_created, sent_dt_created, conversation_id)
        except Exception:
            logger.error(f"Failed to link chat message sent at {sent_dt_created} to chat message received at "
                         f"{received_dt_created}, from conversation {conversation_id}: {format_exc()}")

    async def link_chat_message_to_summary(
            self, conversation_id: int, dt_created: datetime, position: int, summary_text: str
    ):
        try:
            async with self.driver.session() as session:
                await session.execute_write(_link_chat_message_to_summary,
                                            conversation_id, dt_created, position, summary_text)
        except Exception:
            logger.error(f"Failed to link chat message from conversation {conversation_id}, "
                         f"received at {dt_created} to summary, \"{summary_text}\": {format_exc()}")

    async def link_chat_user_to_conversation(self, user_id: int, conversation_id: int):
        try:
            async with self.driver.session() as session:
                await session.execute_write(_link_chat_user_to_conversation,
                                            user_id=user_id, conversation_id=conversation_id)
        except Exception:
            logger.error(f"Failed to link chat user {user_id} to conversation {conversation_id}: {format_exc()}")

    async def match_bot_message_summaries_by_similarity_to_query_vector(
            self, query_vector: list[float], k: int = 5, min_similarity: float = 0.8
    ):
        try:
            async with self.driver.session() as session:
                return await session.execute_read(
                    _match_bot_message_summaries_by_similarity_to_query_vector,
                    query_vector, k=k, min_similarity=min_similarity
                )
        except Exception:
            logger.error(f"Error while fetching bot message summaries: {format_exc()}")
            return []

    async def match_keyword_phrases_by_similarity(
            self, query_vector: list[float],
            k: int = 5, min_similarity: float = 0.8
    ):
        try:
            async with self.driver.session() as session:
                return await session.execute_read(
                    _match_keyword_phrases_by_similarity,
                    query_vector, k=k, min_similarity=min_similarity
                )
        except Exception:
            logger.error(f"Error while fetching search keywords: {format_exc()}")
            return []

    async def match_keyword_phrases_by_similarity_to_query_vector(
            self, query_vector: list[float],
            similar_message_structs: list[dict[str, float | datetime | int]],
            alpha: float = 0.7, k: int = 5,
    ):
        try:
            async with self.driver.session() as session:
                return await session.execute_read(
                    _match_keyword_phrases_by_similarity_to_query_vector, query_vector, similar_message_structs,
                    alpha=alpha, k=k
                )
        except Exception:
            logger.error(f"Error while fetching search keywords: {format_exc()}")
            return []

    async def match_keyword_phrases_by_text(self, texts: list[str]):
        try:
            async with self.driver.session() as session:
                return await session.execute_read(_match_keyword_phrases_by_text, texts=texts)
        except Exception:
            logger.error(f"Error while fetching search keywords: {format_exc()}")
            return []

    async def match_user_message_summaries_by_similarity_to_query_vector(
            self, user_id: int, query_vector: list[float], k: int = 5, min_similarity: float = 0.8
    ):
        try:
            async with self.driver.session() as session:
                return await session.execute_read(
                    _match_user_message_summaries_by_similarity_to_query_vector, user_id, query_vector,
                    k=k, min_similarity=min_similarity
                )
        except Exception:
            logger.error(f"Error while fetching user summaries: {format_exc()}")
            return []

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
