from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Path
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
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from starlette.middleware.base import RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response
from typing import Optional
import hashlib
import os
import subprocess

from api.models import ChatBookmark, ChatRequest, SqlRequest
from utils.openai_utils import do_on_text
from bot.v1 import chat as v1_chat
from bot.v2 import chat as v2_chat
from db.models import (DATABASE_URL, SessionLocal,
                       MessageBookmark, MessageReceived, MessageReplied, engine)
from observability.logging import logging, setup_logging, traceback

resource = Resource.create({
    "service.name": os.getenv("SERVICE_NAME", "germ-bot"),
})

##
# Logging

setup_logging()
logger = logging.getLogger(__name__)

##
# Tracing

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
# App


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Started")
    yield
    logger.info(f"Stopping")


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
# Endpoints


@bot.middleware("http")
async def dispatch(request: Request, call_next: RequestResponseEndpoint) -> Response:
    with tracer.start_as_current_span(f"{request.method} {request.url}", kind=SpanKind.SERVER) as span:
        response = await call_next(request)
        span.set_attribute("http.method", request.method)
        span.set_attribute("http.url", str(request.url))
        span.set_attribute("http.status_code", response.status_code)
        return response


@bot.get("/chat/bookmarks")
async def get_chat_bookmarks(is_test: Optional[bool] = True) -> list[ChatBookmark]:
    bookmark_list = []
    with SessionLocal() as session:
        results = session.query(MessageBookmark).where(MessageBookmark.is_test == is_test).all()
        for message_bookmark in results:
            bookmark_list.append(ChatBookmark(
                id=message_bookmark.id,
                message_received_id=message_bookmark.message_received_id,
                message_replied_id=message_bookmark.message_replied_id,
                message_summary=message_bookmark.message_summary.decode("utf-8"),
            ))
    return bookmark_list


@bot.get("/chat/bookmark/{bookmark_id}")
async def get_chat_bookmark(bookmark_id: int):
    try:
        with SessionLocal() as session:
            bookmark_record = session.query(MessageBookmark).where(
                MessageBookmark.id == bookmark_id,
            ).one_or_none()
            if not bookmark_record:
                raise HTTPException(status_code=404,
                                    detail=f"MessageBookmark.id == {bookmark_id} not found")
            message_received_record = session.query(MessageReceived).where(
                MessageReceived.id == bookmark_record.message_received_id,
            ).one_or_none()
            if not message_received_record:
                raise HTTPException(status_code=404,
                                    detail=f"MessageReceived.id == {bookmark_record.message_received_id} not found")
            message_replied_record = session.query(MessageReplied).where(
                MessageReplied.id == bookmark_record.message_replied_id,
            ).one_or_none()
            if not message_replied_record:
                raise HTTPException(status_code=404,
                                    detail=f"MessageReplied.id == {bookmark_record.message_replied_id} not found")
            return {
                "id": bookmark_record.id,
                "message_summary": bookmark_record.message_summary.decode("utf-8"),
                "message_received": {
                    "id": message_received_record.id,
                    "chat_frame": message_received_record.chat_frame,
                    "content": message_received_record.content.decode("utf-8"),
                },
                "message_replied": {
                    "id": message_replied_record.id,
                    "content": message_replied_record.content.decode("utf-8"),
                    "role": message_replied_record.role,
                },
            }
    except Exception as e:
        logger.error("%s: trace: %s", e, traceback.format_exc())
        raise e if isinstance(e, HTTPException) else HTTPException(status_code=500, detail=str(e))


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


@bot.get("/postgres.html", include_in_schema=False)
async def get_postgres_landing(request: Request):
    file_path = os.path.join(os.path.dirname(__file__), 'static', 'postgres.html')
    return FileResponse(file_path)


@bot.post("/chat")
async def post_chat(payload: ChatRequest, bot_version: str = "v1"):
    try:
        response = version_selector(bot_version)(
            payload.messages,
            system_message=payload.system_message,
            temperature=payload.temperature,
        )
        return response
    except Exception as e:
        logger.error("%s: trace: %s", e, traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@bot.post("/chat/bookmark")
async def post_chat_bookmark(payload: ChatBookmark) -> ChatBookmark:
    try:
        if not payload.message_received_id or not payload.message_replied_id:
            raise HTTPException(status_code=400,
                                detail=f"message_received_id:{payload.message_received_id} "
                                       + f"and message_replied_id:{payload.message_replied_id} are required")
        with SessionLocal() as session:
            # If already bookmarked, return existing
            existing_bookmark_record = session.query(MessageBookmark).where(
                MessageBookmark.message_received_id == payload.message_received_id,
                MessageBookmark.message_replied_id == payload.message_replied_id,
            ).one_or_none()
            if existing_bookmark_record:
                return ChatBookmark(
                    id=existing_bookmark_record.id,
                    message_received_id=existing_bookmark_record.message_received_id,
                    message_replied_id=existing_bookmark_record.message_replied_id,
                    message_summary=existing_bookmark_record.message_summary.decode("utf-8"),
                )

            message_received_record = session.query(MessageReceived).where(
                MessageReceived.id == payload.message_received_id
            ).one_or_none()
            if not message_received_record:
                raise HTTPException(status_code=404,
                                    detail=f"MessageReceived.id == {payload.message_received_id} not found")

            message_replied_record = session.query(MessageReplied).where(
                MessageReplied.id == payload.message_replied_id
            ).one_or_none()
            if not message_replied_record:
                raise HTTPException(status_code=404,
                                    detail=f"MessageReplied.id == {payload.message_replied_id} not found")

            # 70 characters is a "guideline" because we don't enforce it with a `max_tokens=`.
            message_summary = do_on_text(
                "Summarize the following in 70 characters or less",
                message_received_record.content.decode('utf-8'), max_tokens=70)
            new_bookmark_record = MessageBookmark(
                is_test=payload.is_test,
                message_received_id=payload.message_received_id,
                message_replied_id=payload.message_replied_id,
                message_summary=message_summary.encode("utf-8"))
            session.add(new_bookmark_record)
            session.commit()
            session.refresh(new_bookmark_record)
            return ChatBookmark(
                id=new_bookmark_record.id,
                is_test=new_bookmark_record.is_test,
                message_received_id=new_bookmark_record.message_received_id,
                message_replied_id=new_bookmark_record.message_replied_id,
                message_summary=new_bookmark_record.message_summary)
    except Exception as e:
        logger.error("%s: trace: %s", e, traceback.format_exc())
        raise e if isinstance(e, HTTPException) else HTTPException(status_code=500, detail=str(e))


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


##
# Functions


def version_selector(version):
    if version == "v1":
        return v1_chat
    elif version == "v2":
        return v2_chat
    else:
        logger.warning("unknown version: %s, defaulting to v1", version)
        return v1_chat
