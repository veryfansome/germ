from datetime import datetime
import asyncio
import logging

from bot.db.neo4j import AsyncNeo4jDriver

logger = logging.getLogger(__name__)


class ControlPlane:
    def __init__(self, driver: AsyncNeo4jDriver):
        self.driver = driver

    async def add_chat_message(self, message_id: int, dt_created: datetime):
        results = await self.driver.query("""
        MERGE (message:ChatMessage {message_id: $message_id, dt_created: $dt_created})
        RETURN message
        """, {
            "message_id": message_id, "dt_created": dt_created,
        })
        if results:
            logger.info(f"MERGE (message:ChatMessage {{message_id: {message_id}, dt_created: {dt_created}}})")
        return results

    async def add_chat_session(self, session_id: int, dt_created: datetime):
        results = await self.driver.query("""
        MERGE (session:ChatSession {session_id: $session_id, dt_created: $dt_created})
        RETURN session
        """, {
            "session_id": session_id, "dt_created": dt_created
        })
        if results:
            logger.info(f"MERGE (session:ChatSession {{session_id: {session_id}, dt_created: {dt_created}}})")
        return results

    async def add_chat_user(self, user_id: int, dt_created: datetime):
        results = await self.driver.query("""
        MERGE (user:ChatUser {user_id: $user_id, dt_created: $dt_created})
        RETURN user
        """, {
            "user_id": user_id, "dt_created": dt_created
        })
        if results:
            logger.info(f"MERGE (user:ChatUser {{user_id: {user_id}, dt_created: {dt_created}}})")
        return results

    async def close(self):
        await self.driver.shutdown()

    async def link_chat_message_received_to_chat_user(self, message_id: int, user_id: int):
        results = await self.driver.query("""
        MATCH (message:ChatMessage {message_id: $message_id}), (user:ChatUser {user_id: $user_id})
        MERGE (user)-[rel:SENT]->(message)
        RETURN rel
        """, {
            "message_id": message_id, "user_id": user_id,
        })
        if results:
            logger.info("MERGE "
                        f"(user:ChatUser {{user_id: {user_id}}})-[rel:SENT]->"
                        f"(message:ChatMessage {{message_id: {message_id}}})")
        return results

    async def link_chat_message_sent_to_chat_message_received(
            self, received_message_id: int, sent_message_id: int, session_id: int):
        results = await self.driver.query("""
        MATCH (received:ChatMessage {message_id: $received_message_id}), (sent:ChatMessage {message_id: $sent_message_id})
        MERGE (sent)-[rel:REACTS_TO {session_id: $session_id}]->(received)
        RETURN rel
        """, {
            "received_message_id": received_message_id, "sent_message_id": sent_message_id,
            "session_id": session_id
        })
        if results:
            logger.info("MERGE "
                        f"(sent:ChatMessage {{sent_message_id: {sent_message_id}}})-[rel:REACTS_TO]->"
                        f"(received:ChatMessage {{received_message_id: {received_message_id}}})")
        return results

    async def link_chat_user_to_chat_session(self, user_id: int, session_id: int):
        results = await self.driver.query("""
        MATCH (user:ChatUser {user_id: $user_id}), (session:ChatSession {session_id: $session_id})
        MERGE (session)-[rel:WITH]->(user)
        RETURN rel
        """, {
            "user_id": user_id, "session_id": session_id,

        })
        if results:
            logger.info("MERGE "
                        f"(session:ChatSession {{session_id: {session_id}}})-[rel:WITH]->"
                        f"(user:ChatUser {{user_id: {user_id}}})")
        return results


if __name__ == "__main__":
    from observability.logging import setup_logging
    setup_logging()

    async def main():
        neo4j_driver = AsyncNeo4jDriver()
        control_plane = ControlPlane(neo4j_driver)

        await neo4j_driver.shutdown()
    asyncio.run(main())
