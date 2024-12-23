import os
from neo4j import AsyncGraphDatabase, GraphDatabase

from observability.logging import logging, setup_logging
from settings.germ_settings import NEO4J_HOST

logger = logging.getLogger(__name__)

NEO4J_AUTH = os.getenv("NEO4J_AUTH", "neo4j/oops")
neo4j_auth_parts = NEO4J_AUTH.split("/")


class Neo4jDriver:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            f"bolt://{NEO4J_HOST}:7687", auth=(neo4j_auth_parts[0], neo4j_auth_parts[1]))

    def close(self):
        self.driver.close()

    def query(self, query, parameters=None):
        with self.driver.session() as session:
            result = session.run(query, parameters)
            return [record for record in result]

    def delete_all_data(self):
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")


class AsyncNeo4jDriver:
    def __init__(self):
        self.driver = AsyncGraphDatabase.driver(
            f"bolt://{NEO4J_HOST}:7687", auth=(neo4j_auth_parts[0], neo4j_auth_parts[1]))

    def close(self):
        self.driver.close()

    async def query(self, query, parameters=None):
        async with self.driver.session() as session:
            result = await session.run(query, parameters)
            return await result.data()

    async def delete_all_data(self):
        async with self.driver.session() as session:
            result = await session.run("MATCH (n) DETACH DELETE n")
            await result.consume()
            logger.info("Deleted all data")
