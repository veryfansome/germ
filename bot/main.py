from apscheduler.schedulers.asyncio import AsyncIOScheduler
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, HTTPException, Path, UploadFile, WebSocket
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from opentelemetry import trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
from typing import List
import aiofiles
import asyncio
import hashlib
import os
import subprocess
import traceback

from bot.api.models import ChatMessage, ChatSessionSummary, SqlRequest
from bot.chat.controller import ChatController
from bot.chat.openai_beta import AssistantHelper
from bot.chat.openai_handlers import ChatRoutingEventHandler, UserProfilingHandler
from bot.db.models import DATABASE_URL, SessionLocal, engine
from bot.db.neo4j import AsyncNeo4jDriver
from bot.db.utils import db_stats_job
from bot.graph.control_plane import ControlPlane
from bot.lang.controllers.english import EnglishController
from bot.websocket import (WebSocketConnectionManager,
                           get_chat_session_messages, get_chat_session_summaries,
                           update_chat_session_is_hidden)
from observability.logging import logging, setup_logging
from observability.tracing import setup_tracing
from settings import germ_settings

scheduler = AsyncIOScheduler()

##
# Logging

setup_logging()
logger = logging.getLogger(__name__)

##
# Tracing

setup_tracing("bot-service")
tracer = trace.get_tracer(__name__)

##
# App

neo4j_driver = AsyncNeo4jDriver()
control_plane = ControlPlane(neo4j_driver)
websocket_manager = WebSocketConnectionManager(control_plane)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Service startup/shutdown behavior.

    :param app:
    :return:
    """

    # Graph controllers
    english_controller = EnglishController(control_plane)
    await english_controller.initialize()

    control_plane.add_code_block_merge_event_handler(english_controller)
    control_plane.add_paragraph_merge_event_handler(english_controller)
    control_plane.add_sentence_merge_event_handler(english_controller)
    await control_plane.initialize()
    scheduler.add_job(english_controller.label_sentences_periodically, "interval",
                      seconds=english_controller.interval_seconds)

    assistant_helper = AssistantHelper()
    await assistant_helper.refresh_assistants()
    await assistant_helper.refresh_files()
    await assistant_helper.no_loose_files()
    scheduler.scheduled_job(assistant_helper.refresh_files, "interval", minutes=5)

    router = ChatRoutingEventHandler(assistant_helper=assistant_helper)
    chat_controller = ChatController(control_plane, router)

    websocket_manager.add_receive_event_handler(chat_controller)
    websocket_manager.add_send_event_handler(chat_controller)
    websocket_manager.add_session_monitor(chat_controller)

    #user_intent_profiler = UserProfilingHandler(
    #    control_plane,
    #    {
    #        "intent": {
    #            "type": "string",
    #            "description": "A statement beginning with \"The User \", describing what the User wants.",
    #        },
    #    },
    #    post_func=lambda intent: intent if intent.startswith("The User") else f"The User {intent}"  # Needed sometimes
    #)
    #websocket_manager.add_receive_event_handler(user_intent_profiler)

    # DB stats
    await db_stats_job()  # Warms up DB connections on startup
    scheduler.add_job(db_stats_job, "interval", minutes=5)

    # Scheduler
    scheduler.start()

    # Started

    yield

    # Stopping

    websocket_manager_disconnect_task = asyncio.create_task(websocket_manager.disconnect_all())
    await assistant_helper.no_loose_files()
    await assistant_helper.no_loose_threads()
    await neo4j_driver.shutdown()
    await websocket_manager_disconnect_task
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
# Endpoints


@bot.delete("/chat/session/{chat_session_id}")
async def delete_chat_session_bookmark(chat_session_id: int):
    return await update_chat_session_is_hidden(chat_session_id)


@bot.get("/", include_in_schema=False)
async def get_landing():
    file_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    return FileResponse(file_path)


@bot.get("/chat/sessions")
async def get_chat_session_list() -> list[ChatSessionSummary]:
    return await get_chat_session_summaries()


@bot.get("/chat/session/{chat_session_id}")
async def get_chat_session_message_list(chat_session_id: int) -> list[ChatMessage]:
    return await get_chat_session_messages(chat_session_id)


@bot.get("/favicon.ico", include_in_schema=False)
async def get_favicon():
    file_path = os.path.join(os.path.dirname(__file__), "static", "assets", "favicon.ico")
    return FileResponse(file_path)


@bot.get("/graph")
async def get_graph():
    node_results, edge_results = await asyncio.gather(control_plane.get_nodes(), control_plane.get_edges())
    combined_results = {
        "nodes": [],
        "edges": [],
    }
    for r in edge_results:
        edge = {"id": r["edgeId"], "from": r["startNodeId"], "to": r["endNodeId"], "label": r["edge"][1]}
        combined_results["edges"].append(edge)
    for r in node_results:
        node = {"id": r["nodeId"], **r["node"], "nodeLabels": r["nodeLabels"]}
        combined_results["nodes"].append(node)
    return combined_results


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


@bot.post("/upload")
async def post_upload(files: List[UploadFile] = File(...)):
    saved_files = []
    for file in files:
        save_path = os.path.join(germ_settings.UPLOAD_FOLDER, file.filename)
        try:
            async with aiofiles.open(save_path, "wb") as f:
                chunk_size = 1024 * 1024
                while True:
                    chunk = await file.read(chunk_size)
                    if not chunk:
                        break
                    await f.write(chunk)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        saved_files.append(file.filename)
    return {"files_saved": saved_files}


@bot.websocket("/chat")
async def websocket_chat(ws: WebSocket):
    chat_session_id = await websocket_manager.connect(ws)
    logger.info(f"starting session {chat_session_id}")
    await websocket_manager.conduct_chat_session(chat_session_id)
