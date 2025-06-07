import os
import uuid

# Service

ENCRYPTION_PASSWORD = os.getenv("ENCRYPTION_KEY", "0Bfusc8")
LOG_DIR = os.getenv("LOG_DIR", "/var/log/germ")
MODEL_SERVICE_ENDPOINT = os.getenv("MODEL_SERVICE_HOST", "germ-models:9000")
UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "/tmp")
UUID5_NS = uuid.UUID("246a5463-afae-4571-a6e0-f319d74147d3")
WEBSOCKET_IDLE_TIMEOUT = 3600
WEBSOCKET_MONITOR_INTERVAL_SECONDS = 15.0

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
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "bacteria4life")
POSTGRES_USER = os.getenv("POSTGRES_USER", "germ")

# Redis
REDIS_HOST = os.getenv("REDIS_HOST", "germ-redis")
REDIS_PORT = os.getenv("REDIS_PORT", "6379")