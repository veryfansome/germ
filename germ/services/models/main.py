import asyncio
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from opentelemetry import trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from sentence_transformers import SentenceTransformer
from starlette.concurrency import run_in_threadpool
from starlette.responses import Response

from germ.api.models import EmbeddingRequestPayload, TextListPayload
from germ.observability.logging import logging, setup_logging
from germ.observability.tracing import setup_tracing

##
# Logging

setup_logging()
logger = logging.getLogger(__name__)

##
# Tracing

setup_tracing("model-service")
tracer = trace.get_tracer(__name__)

##
# App

text_embedding_model = SentenceTransformer('Snowflake/snowflake-arctic-embed-l-v2.0')
text_embedding_model_dim = text_embedding_model.get_sentence_embedding_dimension()

max_encoding_threads = max(2, os.cpu_count() - 2)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Service startup/shutdown behavior.

    :param app:
    :return:
    """
    # Started
    logger.info("Starting")

    await post_text_embedding(EmbeddingRequestPayload(texts=["Hello, world!"], prompt="passage: "))

    logger.info("Started")

    yield

    logger.info("Stopping")

    # Stopping

    logger.info("Stopped")


model_service = FastAPI(lifespan=lifespan)


##
# Enabled instrumentation
FastAPIInstrumentor.instrument_app(model_service)


##
# Endpoints


@model_service.get("/healthz")
async def get_healthz():
    return {
        "environ": os.environ,
        "status": "OK",
    }


@model_service.get("/metrics")
async def get_metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@model_service.get("/text/embedding/info")
async def get_text_embedding_info():
    return {"dim": text_embedding_model_dim}


@model_service.post("/text/embedding")
async def post_text_embedding(payload: EmbeddingRequestPayload):
    embs = []
    tasks = []
    texts_len = len(payload.texts)
    partition_len = max(1, texts_len // max_encoding_threads)
    for idx in range(0, texts_len, partition_len):
        stop_idx = idx + partition_len
        tasks.append(asyncio.create_task(run_in_threadpool(
            text_embedding_model.encode, payload.texts[idx:stop_idx],
            prompt_name="query" if payload.prompt == "query: " else None,
            normalize_embeddings=False, show_progress_bar=False,
        )))
    for batch_embs in await asyncio.gather(*tasks):
        embs.extend(batch_embs.tolist())
    return {"embeddings": embs}
