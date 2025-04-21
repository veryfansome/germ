from datetime import datetime, timezone
import asyncio
import logging

from bot.db.neo4j import AsyncNeo4jDriver

logger = logging.getLogger(__name__)


class ControlPlane:
    def __init__(self, driver: AsyncNeo4jDriver):
        self.driver = driver

    async def add_chat_request(self, chat_request_received_id: int):
        time_occurred = round_time_now_down_to_nearst_interval()
        results = await self.driver.query("""
        MERGE (chatRequest:ChatRequest {chat_request_received_id: $chat_request_received_id, time_occurred: $time_occurred})
        RETURN chatRequest
        """, {
            "chat_request_received_id": chat_request_received_id, "time_occurred": time_occurred,
        })
        if results:
            logger.info(f"MERGE (req:ChatRequest {{chat_request_received_id: {chat_request_received_id}}})")
        return results, time_occurred

    async def add_chat_response(self, chat_response_sent_id: int):
        time_occurred = round_time_now_down_to_nearst_interval()
        results = await self.driver.query("""
        MERGE (chatResponse:ChatResponse {chat_response_sent_id: $chat_response_sent_id, time_occurred: $time_occurred})
        RETURN chatResponse
        """, {
            "chat_response_sent_id": chat_response_sent_id, "time_occurred": time_occurred,
        })
        if results:
            logger.info(f"MERGE (resp:ChatResponse {{$chat_response_sent_id: {chat_response_sent_id}}})")
        return results, time_occurred

    async def add_chat_session(self, chat_session_id: int):
        results = await self.driver.query("""
        MERGE (session:ChatSession {chat_session_id: $chat_session_id, time_started: $time_started})
        RETURN session
        """, {
            "chat_session_id": chat_session_id, "time_started": round_time_now_down_to_nearst_interval()
        })
        if results:
            logger.info(f"MERGE (session:ChatResponse {{chat_session_id: {chat_session_id}}})")
        return results

    async def close(self):
        await self.driver.shutdown()

    async def get_edges(self):
        return await self.driver.query("""
        MATCH (start)-[edge]->(end) WHERE NOT start:__Neo4jMigration RETURN edge, id(edge) AS edgeId, id(start) AS startNodeId, id(end) AS endNodeId
        """)

    async def get_nodes(self):
        return await self.driver.query("""
        MATCH (node) WHERE NOT node:__Neo4jMigration RETURN node, id(node) AS nodeId, labels(node) AS nodeLabels
        """)

    async def link_chat_request_to_chat_session(self, chat_request_received_id: int, chat_session_id: int, time_occurred):
        results = await self.driver.query("""
        MATCH (req:ChatRequest {chat_request_received_id: $chat_request_received_id}), (session:ChatSession {chat_session_id: $chat_session_id})
        MERGE (session)-[r:RECEIVED {time_occurred: $time_occurred}]->(req)
        RETURN r
        """, {
            "chat_request_received_id": chat_request_received_id, "chat_session_id": chat_session_id,
            "time_occurred": time_occurred,

        })
        if results:
            logger.info("MERGE "
                        f"(session:ChatSession {{chat_session_id: {chat_session_id}}})-[r:RECEIVED]->"
                        f"(req:ChatRequest {{chat_request_received_id: {chat_request_received_id}}})")
        return results

    async def link_chat_response_to_chat_request(self, chat_request_received_id: int, chat_response_sent_id: int,
                                                 chat_session_id: int, time_occurred):
        results = await self.driver.query("""
        MATCH (req:ChatRequest {chat_request_received_id: $chat_request_received_id}), (resp:ChatResponse {chat_response_sent_id: $chat_response_sent_id})
        MERGE (resp)-[r:REACTS_TO {chat_session_id: $chat_session_id, time_occurred: $time_occurred}]->(req)
        RETURN r
        """, {
            "chat_request_received_id": chat_request_received_id, "chat_response_sent_id": chat_response_sent_id,
            "chat_session_id": chat_session_id, "time_occurred": time_occurred
        })
        if results:
            logger.info("MERGE "
                        f"(resp:ChatResponse {{chat_session_id: {chat_session_id}}})-[r:REACTS_TO]->"
                        f"(req:ChatRequest {{chat_response_sent_id: {chat_response_sent_id}}})")
        return results

    async def link_chat_response_to_chat_session(self, chat_response_sent_id: int, chat_session_id: int, time_occurred):
        results = await self.driver.query("""
        MATCH (resp:ChatResponse {chat_response_sent_id: $chat_response_sent_id}), (session:ChatSession {chat_session_id: $chat_session_id})
        MERGE (session)-[r:SENT {time_occurred: $time_occurred}]->(resp)
        RETURN r
        """, {
            "chat_response_sent_id": chat_response_sent_id, "chat_session_id": chat_session_id,
            "time_occurred": time_occurred,

        })
        if results:
            logger.info("MERGE "
                        f"(session:ChatSession {{chat_session_id: {chat_session_id}}})-[r:SENT]->"
                        f"(resp:ChatResponse {{chat_response_sent_id: {chat_response_sent_id}}})")
        return results


def round_time_now_down_to_nearst_interval(interval_minutes: int = 5):
    now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    minutes = now.minute
    remainder = minutes % interval_minutes  # How many interval_minutes increments have passed
    rounded_minutes = minutes - remainder  # Subtract to round down
    return now.replace(minute=rounded_minutes, second=0, microsecond=0)


def utc_now():
    return datetime.now(timezone.utc)


if __name__ == "__main__":
    from observability.logging import setup_logging
    setup_logging()

    async def main():
        neo4j_driver = AsyncNeo4jDriver()
        control_plane = ControlPlane(neo4j_driver)

        await neo4j_driver.shutdown()
    asyncio.run(main())
