import os
import uuid

# Service

DATA_DIR = os.getenv("DATA_DIR", "/src/data/germ")
UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "/tmp")
UUID5_NS = uuid.UUID("246a5463-afae-4571-a6e0-f319d74147d3")  # Changes sentences signatures
WEBSOCKET_CONNECTION_IDLE_TIMEOUT = 3600

# Observability

JAEGER_HOST = os.getenv("GERM_JAEGER_HOST", "germ-jaeger")
JAEGER_PORT = os.getenv("GERM_JAEGER_PORT", "4317")
LOG_LEVEL = os.getenv("GERM_LOG_LEVEL", "INFO")

# Neo4j
NEO4J_AUTH = os.getenv("NEO4J_AUTH", "neo4j/oops")
NEO4J_HOST = os.getenv("GERM_NEO4J_HOST", "germ-neo4j")
NEO4J_PORT = os.getenv("GERM_NEO4J_PORT", "7687")

# PostgreSQL

DB_HOST = os.getenv("DB_HOST", "germ-pg")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "germ")
POSTGRES_USER = os.getenv("POSTGRES_USER", "bacteria4life")
