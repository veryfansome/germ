from datetime import datetime
import asyncio
import logging

from germ.bot.db.neo4j import AsyncNeo4jDriver

logger = logging.getLogger(__name__)


class ControlPlane:
    def __init__(self, driver: AsyncNeo4jDriver):
        self.driver = driver

    async def add_chat_message(self, conversation_id: int, dt_created: datetime):
        results = await self.driver.query("""
        MERGE (message:ChatMessage {conversation_id: $conversation_id, dt_created: $dt_created})
        RETURN message
        """, {
            "conversation_id": conversation_id, "dt_created": dt_created,
        })
        if results:
            logger.info(f"MERGE (message:ChatMessage {{conversation_id: {conversation_id}, dt_created: {dt_created}}})")
        return results

    async def add_conversation(self, conversation_id: int, dt_created: datetime):
        results = await self.driver.query("""
        MERGE (conversation:Conversation {conversation_id: $conversation_id, dt_created: $dt_created})
        RETURN conversation
        """, {
            "conversation_id": conversation_id, "dt_created": dt_created
        })
        if results:
            logger.info(f"MERGE (conversation:Conversation {{conversation_id: {conversation_id}, dt_created: {dt_created}}})")
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

    async def link_chat_message_received_to_chat_user(self, conversation_id: int, dt_created: datetime, user_id: int):
        results = await self.driver.query("""
        MATCH (message:ChatMessage {conversation_id: $conversation_id, dt_created: $dt_created}), (user:ChatUser {user_id: $user_id})
        MERGE (user)-[rel:SENT]->(message)
        RETURN rel
        """, {
            "conversation_id": conversation_id, "dt_created": dt_created, "user_id": user_id,
        })
        if results:
            logger.info("MERGE "
                        f"(user:ChatUser {{user_id: {user_id}}})-[rel:SENT]->"
                        f"(message:ChatMessage {{conversation_id: {conversation_id}, dt_created: {dt_created}}})")
        return results

    async def link_chat_message_sent_to_chat_message_received(
            self, received_dt_created: datetime, sent_dt_created: datetime, conversation_id: int):
        results = await self.driver.query("""
        MATCH (received:ChatMessage {conversation_id: $conversation_id, dt_created: $received_dt_created}), (sent:ChatMessage {conversation_id: $conversation_id, dt_created: $sent_dt_created})
        MERGE (sent)-[rel:REACTS_TO]->(received)
        RETURN rel
        """, {
            "received_dt_created": received_dt_created, "sent_dt_created": sent_dt_created,
            "conversation_id": conversation_id
        })
        if results:
            logger.info("MERGE "
                        f"(sent:ChatMessage {{conversation_id: {conversation_id}, dt_created: {sent_dt_created}}})-[rel:REACTS_TO]->"
                        f"(received:ChatMessage {{conversation_id: {conversation_id}, dt_created: {received_dt_created}}})")
        return results

    async def link_chat_user_to_conversation(self, user_id: int, conversation_id: int):
        results = await self.driver.query("""
        MATCH (user:ChatUser {user_id: $user_id}), (conversation:Conversation {conversation_id: $conversation_id})
        MERGE (conversation)-[rel:WITH]->(user)
        RETURN rel
        """, {
            "user_id": user_id, "conversation_id": conversation_id,

        })
        if results:
            logger.info("MERGE "
                        f"(conversation:Conversation {{conversation_id: {conversation_id}}})-[rel:WITH]->"
                        f"(user:ChatUser {{user_id: {user_id}}})")
        return results


if __name__ == "__main__":
    from germ.observability.logging import setup_logging
    setup_logging()

    async def main():
        neo4j_driver = AsyncNeo4jDriver()
        control_plane = ControlPlane(neo4j_driver)

        await neo4j_driver.shutdown()
    asyncio.run(main())
