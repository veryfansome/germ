from contextlib import asynccontextmanager
from fastapi import FastAPI
from opentelemetry import trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from sentence_transformers import SentenceTransformer
from starlette.concurrency import run_in_threadpool
from starlette.responses import Response
import os

from germ.api.models import EmbeddingRequestPayload, TextListPayload
from germ.observability.logging import logging, setup_logging
from germ.observability.tracing import setup_tracing
from germ.services.models.predict.multi_predict import MultiHeadPredictor

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

text_embedding_model = SentenceTransformer('intfloat/e5-base-v2')
text_embedding_model_dim = text_embedding_model.get_sentence_embedding_dimension()
ud_token_multi_classifier = MultiHeadPredictor(
    "veryfansome/multi-classifier", subfolder="models/ud_ewt_gum_pud_20250611")


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
    await post_text_classification_ud(TextListPayload(texts=["Hello, world!"]))

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


@model_service.post("/text/classification/ud")
async def post_text_classification_ud(payload: TextListPayload):
    return await run_in_threadpool(ud_token_multi_classifier.predict_batch, payload.texts)


@model_service.post("/text/embedding")
async def post_text_embedding(payload: EmbeddingRequestPayload):
    embeddings = await run_in_threadpool(text_embedding_model.encode, payload.texts,
                                         prompt=payload.prompt, normalize_embeddings=False, show_progress_bar=False)
    return {"embeddings": embeddings.tolist()}
