import os

from transformers.utils.constants import OPENAI_CLIP_STD, OPENAI_CLIP_MEAN

# Service

ENCRYPTION_PASSWORD = os.getenv("ENCRYPTION_KEY", "0Bfusc8")
LOG_DIR = os.getenv("LOG_DIR", "/var/log/germ")
MESSAGE_LOG_FILENAME = os.getenv("MESSAGE_LOG_FILENAME", "message.log")
MODEL_SERVICE_ENDPOINT = os.getenv("MODEL_SERVICE_HOST", "germ-models:9000")
UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "/tmp")
WEBSOCKET_IDLE_TIMEOUT = 3600
WEBSOCKET_MONITOR_INTERVAL_SECONDS = 300.0

# Observability

JAEGER_HOST = os.getenv("GERM_JAEGER_HOST", "germ-jaeger")
JAEGER_PORT = os.getenv("GERM_JAEGER_PORT", "4317")
LOG_LEVEL = os.getenv("GERM_LOG_LEVEL", "INFO")

# Neo4j
NEO4J_AUTH = os.getenv("NEO4J_AUTH", "neo4j/oops")
NEO4J_HOST = os.getenv("GERM_NEO4J_HOST", "germ-neo4j")
NEO4J_PORT = os.getenv("GERM_NEO4J_PORT", "7687")

# OpenAI
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o")
OPENAI_CLASSIFICATION_MODEL = os.getenv("OPENAI_CLASSIFICATION_MODEL", "gpt-4o-mini")
OPENAI_CURATION_MODEL = os.getenv("OPENAI_CURATION_MODEL", "gpt-4o-mini")
OPENAI_DEDUP_MODEL = os.getenv("OPENAI_DEDUP_MODEL", "gpt-4o-mini")
OPENAI_REASONING_MODEL = os.getenv("OPENAI_REASONING_MODEL")
OPENAI_RELEVANCE_GATE_MODEL = os.getenv("OPENAI_RELEVANCE_GATE_MODEL", "gpt-4o")
OPENAI_SEED = int(os.getenv("OPENAI_SEED", "1234"))
OPENAI_SUMMARY_MODEL = os.getenv("OPENAI_SUMMARY_MODEL", "gpt-4o-mini")

# PostgreSQL

DB_HOST = os.getenv("DB_HOST", "germ-pg")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "bacteria4life")
POSTGRES_USER = os.getenv("POSTGRES_USER", "germ")

# Redis
REDIS_HOST = os.getenv("REDIS_HOST", "germ-redis")
REDIS_PORT = os.getenv("REDIS_PORT", "6379")