import asyncio
import os
import time
import uuid
from neo4j import AsyncResult, AsyncGraphDatabase, GraphDatabase

from observability.logging import logging
from settings.germ_settings import NEO4J_HOST, UUID5_NS

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
        self.query_cache: dict[str, tuple[float, list]] = {}

    def close(self):
        self.driver.close()

    async def query(self, query, parameters=None):
        query_signature = str(uuid.uuid5(
            UUID5_NS, query + ''.join([str(pv) for pv in (parameters if parameters is not None else {}).values()])))
        if query_signature not in self.query_cache:
            self.query_cache[query_signature] = (time.time(), [])
        elif time.time() - self.query_cache[query_signature][0] < 0.1:
            cnt = 0
            max_cnt = 3
            while cnt < max_cnt:
                cnt += 1
                if not self.query_cache[query_signature][1]:
                    delay_seconds = 0.5 * cnt
                    logger.info(f"will check again for result after {delay_seconds}s: {query}, {parameters}")
                    await asyncio.sleep(delay_seconds)
            return self.query_cache[query_signature][1]
        async with self.driver.session() as session:
            result = await session.run(query, parameters)
            records = await result.data()
            self.query_cache[query_signature] = (time.time(), records)
            return records

    async def delete_all_data(self):
        async with self.driver.session() as session:
            result = await session.run("MATCH (n) DETACH DELETE n")
            await result.consume()
            logger.info("Deleted all data")
