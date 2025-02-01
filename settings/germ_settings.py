import os
import uuid

# Service

SERVICE_NAME = os.getenv("GERM_SERVICE_NAME", "germ-bot")
WEBSOCKET_CONNECTION_IDLE_TIMEOUT = 3600

UUID5_NS = uuid.UUID("246a5463-afae-4571-a6e0-f319d74147d3")  # Changes sentences signatures

# Model training

DEFAULT_BERT_MODEL = os.getenv("GERM_DEFAULT_BERT_MODEL", "distilbert-base-cased")
IMAGE_MODEL_STARTERKIT_TRAINING_ROUNDS = os.getenv("GERM_IMAGE_MODEL_STARTERKIT_TRAINING_ROUNDS", 50)
MODEL_DIR = os.getenv("GERM_MODEL_DIR", "/var/lib/germ/models")
STARTERKIT_DIR = os.getenv("GERM_STARTERKIT_DIR", "/src/data/germ/starterkit")

# Observability

JAEGER_HOST = os.getenv("GERM_JAEGER_HOST", "germ-jaeger")
JAEGER_PORT = os.getenv("GERM_JAEGER_PORT", "6831")
LOG_LEVEL = os.getenv("GERM_LOG_LEVEL", "INFO")

# Neo4j
NEO4J_AUTH = os.getenv("NEO4J_AUTH", "neo4j/oops")
NEO4J_HOST = os.getenv("GERM_NEO4J_HOST", "germ-neo4j")
NEO4J_PORT = os.getenv("GERM_NEO4J_PORT", "7687")

# PostgreSQL

DB_HOST = os.getenv("DB_HOST", "germ-db")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "germ")
POSTGRES_USER = os.getenv("POSTGRES_USER", "bacteria4life")
