import os
from neo4j import GraphDatabase

NEO4J_AUTH = os.getenv("NEO4J_AUTH", "neo4j/oops")
neo4j_auth_parts = NEO4J_AUTH.split("/")


class Neo4jDriver:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            "bolt://germ-neo4j:7687", auth=(neo4j_auth_parts[0], neo4j_auth_parts[1]))

    def close(self):
        self.driver.close()

    def query(self, query, parameters=None):
        with self.driver.session() as session:
            result = session.run(query, parameters)
            return [record for record in result]

    def delete_all_data(self):
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
