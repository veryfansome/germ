from contextlib import asynccontextmanager
from fastapi import FastAPI
from opentelemetry import trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from starlette.concurrency import run_in_threadpool
from starlette.responses import Response
import asyncio
import os

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
