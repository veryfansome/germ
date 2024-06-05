from apscheduler.schedulers.asyncio import AsyncIOScheduler
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Path, WebSocket
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import SpanKind
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from starlette.concurrency import run_in_threadpool
from starlette.middleware.base import RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response
import asyncio
import hashlib
import os
import subprocess
import traceback

from bot.api.models import ChatMessage, ChatSessionSummary, SqlRequest
from bot.chat.openai_handlers import ChatRoutingEventHandler, ResponseGraphingHandler, UserProfilingHandler
from bot.db.models import DATABASE_URL, SessionLocal, engine
from bot.db.neo4j import AsyncNeo4jDriver
from bot.db.utils import db_stats_job
from bot.graph.control_plane import ControlPlane
from bot.lang.controllers.english import EnglishController
from bot.websocket import (WebSocketConnectionManager,
                           get_chat_session_messages, get_chat_session_summaries,
                           update_chat_session_is_hidden)
from observability.logging import logging, setup_logging
from settings import germ_settings

scheduler = AsyncIOScheduler()

##
# Logging

setup_logging()
logger = logging.getLogger(__name__)

##
# Tracing

resource = Resource.create({
    "service.name": germ_settings.SERVICE_NAME,
})
provider = TracerProvider(resource=resource)
trace.set_tracer_provider(provider)

otlp_exporter = OTLPSpanExporter(
    endpoint=f"{germ_settings.JAEGER_HOST}:{germ_settings.JAEGER_PORT}",
    insecure=True,
)

span_processor = BatchSpanProcessor(otlp_exporter)

provider.add_span_processor(span_processor)
tracer = trace.get_tracer(__name__)

##
# App

neo4j_driver = AsyncNeo4jDriver()

control_plane = ControlPlane(neo4j_driver)

english_controller = EnglishController(control_plane)
control_plane.add_code_block_merge_event_handler(english_controller)
control_plane.add_paragraph_merge_event_handler(english_controller)
control_plane.add_sentence_merge_event_handler(english_controller)

response_grapher = ResponseGraphingHandler(control_plane)

websocket_manager = WebSocketConnectionManager(control_plane)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Service startup/shutdown behavior.

    :param app:
    :return:
    """
    await control_plane.initialize()

    router = ChatRoutingEventHandler()
    # TODO: maybe this shouldn't happen with every message.
    # user_fact_profiler = UserProfilingHandler({
    #    "user_fact": {
    #        "type": "string",
    #        "description": ("Using a statement that beings with \"The User\", infer something about the User "
    #                        "that must be true."),
    #    },
    # })
    user_intent_profiler = UserProfilingHandler(
        control_plane,
        {
            "intent": {
                "type": "string",
                "description": "A statement beginning with \"The User \", describing what the User wants.",
            },
        },
        post_func=lambda intent: intent if intent.startswith("The User") else f"The User {intent}"  # Needed sometimes
    )
    websocket_manager.add_send_event_handler(response_grapher)
    websocket_manager.add_receive_event_handler(router)
    # websocket_manager.add_ws_event_handler(user_fact_profiler)
    websocket_manager.add_receive_event_handler(user_intent_profiler)

    # DB stats
    await db_stats_job()  # Warms up DB connections on startup
    websocket_manager.background_thread.start()
    # Started

    yield

    # Stopping
    websocket_manager_disconnect_task = asyncio.create_task(websocket_manager.disconnect_all())
    await neo4j_driver.shutdown()

    await websocket_manager_disconnect_task
    websocket_manager.background_loop.stop()
    websocket_manager.background_thread.join()
    scheduler.shutdown()


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
    chat_session_id = await websocket_manager.connect(ws)
    logger.info(f"starting session {chat_session_id}")
    await websocket_manager.monitor_chat_session(chat_session_id, ws)
    await websocket_manager.conduct_chat_session(chat_session_id)
