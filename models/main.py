from contextlib import asynccontextmanager
from fastapi import FastAPI
from opentelemetry import trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from starlette.concurrency import run_in_threadpool
from starlette.responses import Response
import os

from bot.api.models import TextPayload
from models.predict.goemotions_predict import GoEmotionsPredictor
from models.predict.multi_predict import MultiHeadPredictor
from observability.logging import logging, setup_logging
from observability.tracing import setup_tracing

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

text_emotions_classifier = GoEmotionsPredictor(
    "veryfansome/deberta-goemotions", subfolder="pos_weight_best")
text_token_multi_classifier = MultiHeadPredictor(
    "veryfansome/multi-classifier", subfolder="models/ud_ewt_gum_20250304")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Service startup/shutdown behavior.

    :param app:
    :return:
    """
    # Started

    yield

    # Stopping


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


@model_service.post("/text/classification/emotions")
async def post_text_classification_emotions(payload: TextPayload):
    return await run_in_threadpool(text_emotions_classifier.predict, [payload.text], use_per_label=True)


@model_service.post("/text/classification/multi")
async def post_text_classification(payload: TextPayload):
    return await run_in_threadpool(text_token_multi_classifier.predict, payload.text)
