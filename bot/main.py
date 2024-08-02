from apscheduler.schedulers.asyncio import AsyncIOScheduler
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Path, WebSocket
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider, SynchronousMultiSpanProcessor
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import SpanKind
from prometheus_client import Gauge, generate_latest, CONTENT_TYPE_LATEST
from starlette.concurrency import run_in_threadpool
from starlette.middleware.base import RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response
from starlette.websockets import WebSocketDisconnect
import asyncio
import hashlib
import os
import subprocess

from api.models import ChatMessage, ChatSessionSummary, SqlRequest
from bot.websocket import (WebSocketConnectionManager,
                           get_chat_session_messages, get_chat_session_summaries,
                           update_chat_session_is_hidden)
from chat.openai_handlers import CHAT_HANDLERS
from db.models import (DATABASE_URL, SessionLocal,
                       ChatSession, ChatRequestReceived, ChatResponseSent,
                       engine)
from observability.logging import logging, setup_logging, traceback

scheduler = AsyncIOScheduler()

##
# Logging

setup_logging()
logger = logging.getLogger(__name__)

##
# Tracing

resource = Resource.create({
    "service.name": os.getenv("SERVICE_NAME", "germ-bot"),
})
multi_span_processor = SynchronousMultiSpanProcessor()
multi_span_processor.add_span_processor(
    BatchSpanProcessor(
        OTLPSpanExporter(
            endpoint="http://{otlp_host}:{otlp_port}/v1/traces".format(
                otlp_host=os.getenv("OTLP_HOST", "germ-otel-collector"),
                otlp_port=os.getenv("OTLP_PORT", "4318")
            ),
        )
    )
)
trace.set_tracer_provider(
    TracerProvider(resource=resource, active_span_processor=multi_span_processor)
)
tracer = trace.get_tracer(__name__)


##
# Metrics

METRIC_CHAT_SESSION_ROW_COUNT = Gauge(
    "chat_session_row_count", "Number of rows in the chat_session table")

METRIC_CHAT_REQUEST_RECEIVED_ROW_COUNT = Gauge(
    "chat_request_received_row_count", "Number of rows in the chat_request_received table")

METRIC_CHAT_RESPONSE_SENT_ROW_COUNT = Gauge(
    "chat_response_sent_row_count", "Number of rows in the chat_response_sent table")


##
# App

websocket_manager = WebSocketConnectionManager()
for model_name, chat_handler in CHAT_HANDLERS.items():
    websocket_manager.add_event_handler(chat_handler)
    logger.info(f"added %s %s", model_name, chat_handler)


@asynccontextmanager
async def lifespan(app: FastAPI):
    async def db_stats_job():
        await run_in_threadpool(db_stats)
    await db_stats_job()  # Warms up DB connections on startup
    scheduler.add_job(db_stats_job, 'interval', seconds=60)
    scheduler.start()
    # Started
    yield
    # Stopping
    # TODO: Debug why I'm not seeing sessions get saved on this kind of disconnect
    await websocket_manager.disconnect_all()

bot = FastAPI(lifespan=lifespan)
bot.mount("/static", StaticFiles(directory="bot/static"), name="static")
if os.path.exists("bot/static/tests"):
    bot.mount("/tests", StaticFiles(directory="bot/static/tests"), name="tests")
    bot.mount("/tests/cov", StaticFiles(directory="bot/static/tests/cov"), name="tests_cov")


##
# Enabled instrumentation
FastAPIInstrumentor.instrument_app(bot)
SQLAlchemyInstrumentor().instrument(engine=engine)


##
# Middleware


@bot.middleware("http")
async def dispatch(request: Request, call_next: RequestResponseEndpoint) -> Response:
    with tracer.start_as_current_span(f"{request.method} {request.url}", kind=SpanKind.SERVER) as span:
        response = await call_next(request)
        span.set_attribute("http.method", request.method)
        span.set_attribute("http.url", str(request.url))
        span.set_attribute("http.status_code", response.status_code)
        return response


##
# Endpoints


@bot.delete("/chat/session/{chat_session_id}")
async def delete_chat_session_bookmark(chat_session_id: int):
    return await run_in_threadpool(update_chat_session_is_hidden, chat_session_id)


@bot.get("/chat/sessions")
async def get_chat_session_list() -> list[ChatSessionSummary]:
    return await run_in_threadpool(get_chat_session_summaries)


@bot.get("/chat/session/{chat_session_id}")
async def get_chat_session_message_list(chat_session_id: int) -> list[ChatMessage]:
    return await run_in_threadpool(get_chat_session_messages, chat_session_id)


@bot.get("/favicon.ico", include_in_schema=False)
async def get_favicon():
    file_path = os.path.join(os.path.dirname(__file__), 'static', 'favicon.ico')
    return FileResponse(file_path)


@bot.get("/", include_in_schema=False)
async def get_landing():
    file_path = os.path.join(os.path.dirname(__file__), 'static', 'index.html')
    return FileResponse(file_path)


@bot.get("/healthz")
async def get_healthz():
    # Is PostgreSQL usable?
    session = SessionLocal()
    session.close()
    # Is OpenAI usable?
    openai_client = OpenAI()
    openai_client.close()
    return {
        "db_url": DATABASE_URL,
        "environ": os.environ,
        "status": "OK",
    }


@bot.get("/metrics")
async def get_metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@bot.post("/postgres/{db}/query")
async def post_postgres_query(payload: SqlRequest,
                              db: str = Path(..., title="Postgres DB")):
    enabled_dbs = {
        "germ": f"postgresql://{DATABASE_URL}"
    }
    if db not in enabled_dbs:
        raise HTTPException(status_code=400, detail="Not supported")

    md5 = hashlib.md5()
    md5.update(payload.sql.encode())
    query_file = f"/tmp/{db}_query_{md5.hexdigest()}.sql"
    with open(query_file, "w") as f:
        f.write(payload.sql)
    try:
        cmd = subprocess.run(f"psql '{enabled_dbs[db]}' -f {query_file}",
                             shell=True, capture_output=True, text=True, check=True)
        os.remove(query_file)
        return f"""**rc**: {cmd.returncode}\n
**stdout**:
```text
{cmd.stdout}
```
**stderr**:
```text
{cmd.stderr}
```
"""
    except Exception as e:
        os.remove(query_file)
        logger.error("%s: trace: %s", e, traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@bot.websocket("/chat")
async def websocket_chat(ws: WebSocket):
    async def conduct_chat_session():
        logger.info("WebSocket connecting")
        chat_session_id = await websocket_manager.connect(ws)
        try:
            while True:
                await websocket_manager.receive(chat_session_id)
        except WebSocketDisconnect:
            logger.info(f"WebSocket for chat_session {chat_session_id} disconnected")
            await websocket_manager.disconnect(chat_session_id)
    await asyncio.create_task(conduct_chat_session())


##
# Functions


def db_stats():
    with SessionLocal() as session:
        chat_session_count = session.query(ChatSession).count()
        METRIC_CHAT_SESSION_ROW_COUNT.set(chat_session_count)
        chat_request_received_count = session.query(ChatRequestReceived).count()
        METRIC_CHAT_REQUEST_RECEIVED_ROW_COUNT.set(chat_request_received_count)
        chat_response_sent_count = session.query(ChatResponseSent).count()
        METRIC_CHAT_RESPONSE_SENT_ROW_COUNT.set(chat_response_sent_count)
        logger.info("db_stats: %s", " ".join((
            f"chat_session_count={chat_session_count}",
            f"chat_request_received_count={chat_request_received_count}",
            F"chat_response_sent_count={chat_response_sent_count}"
        )))
