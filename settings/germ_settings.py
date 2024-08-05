import os

# Service

SERVICE_NAME = os.getenv("GERM_SERVICE_NAME", "germ-bot")

# Model training

DEFAULT_BERT_MODEL = os.getenv("GERM_DEFAULT_BERT_MODEL", "distilbert-base-cased")
IMAGE_MODEL_STARTERKIT_TRAINING_ROUNDS = os.getenv("GERM_IMAGE_MODEL_STARTERKIT_TRAINING_ROUNDS", 10)
MODEL_DIR = os.getenv("GERM_MODEL_DIR", "/var/lib/germ/models")
STARTERKIT_DIR = os.getenv("GERM_STARTERKIT_DIR", "/src/data/germ/starterkit")

# Observability

LOG_LEVEL = os.getenv("GERM_LOG_LEVEL", "INFO")
OTLP_HOST = os.getenv("GERM_OTLP_HOST", "germ-otel-collector")
OTLP_PORT = os.getenv("GERM_OTLP_PORT", "4318")

# PostgreSQL

DB_HOST = os.getenv("DB_HOST", "germ-db")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "germ")
POSTGRES_USER = os.getenv("POSTGRES_USER", "bacteria4life")