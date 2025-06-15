from apscheduler.schedulers.asyncio import AsyncIOScheduler
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket, status
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from opentelemetry import trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from redis.asyncio import Redis
from sqlalchemy import create_engine as create_pg_engine
from sqlalchemy.ext.asyncio import (async_sessionmaker as async_pg_session_maker,
                                    create_async_engine as create_async_pg_engine,
                                    AsyncSession)
from starlette.middleware import Middleware
from starlette.responses import Response
from starsessions import SessionMiddleware, load_session
from starsessions.stores.redis import RedisStore
from typing import List
from urllib.parse import urlencode
import aiofiles
import asyncio
import os

from germ.api.models import ChatResponse
from germ.database.neo4j import KnowledgeGraph
from germ.database.pg import DATABASE_URL, TableHelper
from germ.observability.logging import logging, setup_logging
from germ.observability.tracing import setup_tracing
from germ.services.bot.auth import MAX_COOKIE_AGE, SESSION_COOKIE_NAME, AuthHelper, verify_password
from germ.services.bot.chat.controller import ChatController
from germ.services.bot.chat.openai_beta import AssistantHelper
from germ.services.bot.chat.openai_handlers import ChatRoutingEventHandler
from germ.services.bot.websocket import WebSocketConnectionManager
from germ.settings import germ_settings

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
# Databases

pg_async_engine = create_async_pg_engine(f"postgresql+asyncpg://{DATABASE_URL}", echo=False)
pg_sync_engine = create_pg_engine(f"postgresql+psycopg2://{DATABASE_URL}", echo=False)
pg_session_maker = async_pg_session_maker(
    bind=pg_async_engine,
    class_=AsyncSession,
    expire_on_commit=False
)
pg_table_helper = TableHelper(pg_sync_engine)
redis_client = Redis.from_url(
    f"redis://{germ_settings.REDIS_HOST}:{germ_settings.REDIS_PORT}/0",
    decode_responses=False
)

##
# App

auth_helper = AuthHelper(pg_table_helper.chat_user_table, pg_session_maker)
knowledge_graph = KnowledgeGraph()
websocket_manager = WebSocketConnectionManager(
    pg_table_helper.conversation_table,
    pg_table_helper.conversation_state_table,
    knowledge_graph,
    pg_session_maker,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Service startup/shutdown behavior.

    :param app:
    :return:
    """
    logger.info("Starting")

    assistant_helper = AssistantHelper()
    await assistant_helper.refresh_assistants()
    await assistant_helper.refresh_files()
    await assistant_helper.no_loose_files()
    scheduler.scheduled_job(assistant_helper.refresh_files, "interval", minutes=5)

    router = ChatRoutingEventHandler(assistant_helper=assistant_helper)
    chat_controller = ChatController(knowledge_graph, router)
    await chat_controller.on_start()

    websocket_manager.add_conversation_monitor(chat_controller)
    websocket_manager.add_receive_event_handler(chat_controller)
    websocket_manager.add_send_event_handler(chat_controller)

    # Scheduler
    scheduler.start()

    # Started
    logger.info("Started")

    yield

    # Stopping
    logger.info("Stopping")

    scheduler.shutdown()
    await asyncio.gather(*[
        assistant_helper.no_loose_files(),
        assistant_helper.no_loose_threads(),
        websocket_manager.disconnect_all(),
    ])
    await asyncio.gather(*[
        knowledge_graph.shutdown(),
        pg_async_engine.dispose(),
        redis_client.close(),
    ])
    pg_sync_engine.dispose(),

    logger.info("Stopped")


bot_service = FastAPI(
    lifespan=lifespan,
    middleware=[
        Middleware(
            SessionMiddleware,
            store=RedisStore(connection=redis_client, prefix="sess:"),
            cookie_name=SESSION_COOKIE_NAME,
            cookie_https_only=False,
            lifetime=MAX_COOKIE_AGE,
            rolling=True,
        )
    ]
)
bot_service.mount("/static", StaticFiles(directory="germ/services/bot/static"), name="static")
if os.path.exists("germ/services/bot/static/tests"):
    bot_service.mount("/tests", StaticFiles(directory="germ/services/bot/static/tests"), name="tests")
    bot_service.mount("/tests/cov", StaticFiles(directory="germ/services/bot/static/tests/cov"), name="tests_cov")


##
# Enabled instrumentation
FastAPIInstrumentor.instrument_app(bot_service)
RedisInstrumentor().instrument(client=redis_client)
SQLAlchemyInstrumentor().instrument(engine=pg_async_engine.sync_engine)  # Async is a facade backed by a sync engine


##
# Endpoints


@bot_service.get("/favicon.ico", include_in_schema=False)
async def get_favicon():
    file_path = os.path.join(os.path.dirname(__file__), "static", "assets", "favicon.ico")
    return FileResponse(file_path)


@bot_service.get("/healthz")
async def get_healthz():
    return {
        "environ": os.environ,
        "status": "OK",
    }


@bot_service.get("/", include_in_schema=False)
async def get_landing(request: Request):
    await load_session(request)
    session = request.session
    if not session:
        return RedirectResponse(url="/login", status_code=status.HTTP_303_SEE_OTHER)
    file_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    return FileResponse(
        file_path,
        headers={
            "Cache-Control": "no-store"  # Prevents serving landing from cache after logout
        }
    )


@bot_service.get("/login", include_in_schema=False)
async def get_login_form(request:  Request):
    await load_session(request)
    session = request.session
    if session:
        return RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)
    file_path = os.path.join(os.path.dirname(__file__), "static", "login.html")
    return FileResponse(file_path)


@bot_service.get("/logout", include_in_schema=False)
async def get_logout(request: Request):
    await load_session(request)
    request.session.clear()
    info_params = urlencode({"info": "Logged out."})
    return RedirectResponse(url=f"/login?{info_params}", status_code=status.HTTP_303_SEE_OTHER)


@bot_service.get("/metrics")
async def get_metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@bot_service.get("/register", include_in_schema=False)
async def get_register(request:  Request):
    await load_session(request)
    session = request.session
    if session:
        return RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)
    file_path = os.path.join(os.path.dirname(__file__), "static", "register.html")
    return FileResponse(file_path)


@bot_service.get("/user/info", include_in_schema=False)
async def get_user_info(request:  Request):
    await load_session(request)
    session = request.session
    if not session:
        return RedirectResponse(f"/login", status_code=status.HTTP_303_SEE_OTHER)
    return session


@bot_service.post("/login", include_in_schema=False)
async def post_login_form(request: Request, username: str = Form(...), password: str = Form(...)):
    load_session_task = asyncio.create_task(load_session(request))
    password_hash = await auth_helper.get_user_password_hash(username)
    if password_hash is None:
        error_params = urlencode({"error": "No such user."})
        logger.error(f"No such user: {username}")
    elif not verify_password(password, password_hash):
        error_params = urlencode({"error": "Invalid password."})
        logger.error(f"Invalid password: {username}")
    else:
        user_id = await auth_helper.get_user_id(username)
        await load_session_task
        request.session.clear()
        request.session.update({"user_id": user_id, "username": username})
        logger.info(f"{username} logged in")
        return RedirectResponse("/", status_code=status.HTTP_303_SEE_OTHER)
    return RedirectResponse(f"/login?{error_params}", status_code=status.HTTP_303_SEE_OTHER)


@bot_service.post("/register", include_in_schema=False)
async def post_register(request:  Request, username: str = Form(...), password: str = Form(...), verify: str = Form(...)):
    load_session_task = asyncio.create_task(load_session(request))
    if (await auth_helper.get_user_password_hash(username)) is not None:
        error_params = urlencode({"error": "Username already exists."})
        logger.error(f"Username already exists: {username}")
    elif password != verify:
        error_params = urlencode({"error": "'Password' and 'Verify password' did not match."})
        logger.error(f"Mistyped password: {username}")
    else:
        user_id, dt_created = await auth_helper.add_new_user(username, password)
        if user_id is None:
            error_params = urlencode({"error": "Failed to create new user."})
            logger.error(f"Failed to create new user: {username}")
        else:
            await knowledge_graph.add_chat_user(user_id, dt_created)
            await load_session_task
            request.session.clear()
            request.session.update({"user_id": user_id, "username": username})
            logger.info(f"Registered new user {username}")
            return RedirectResponse("/", status_code=status.HTTP_303_SEE_OTHER)
    return RedirectResponse(f"/register?{error_params}", status_code=status.HTTP_303_SEE_OTHER)


@bot_service.post("/upload")
async def post_upload(request:  Request, files: List[UploadFile] = File(...)):
    await load_session(request)
    session = request.session
    if not session:
        return HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)

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
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
        saved_files.append(file.filename)
    return {"files_saved": saved_files}


@bot_service.websocket("/chat")
async def websocket_chat(ws: WebSocket):
    await ws.accept()
    await load_session(ws)
    session = ws.scope.get("session")
    if not session or "user_id" not in session:
        await ws.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    user_id = session["user_id"]
    if "conversation_ident" in ws.url.query:
        query_params = dict(map(lambda s: s.split('='), ws.url.query.split('&')))
        conversation_id = await websocket_manager.connect(
            user_id, ws, conversation_ident=query_params["conversation_ident"]
        )
        await websocket_manager.new_ws_sender(conversation_id, ws).send_message(
            ChatResponse(complete=True, content="Reconnected!")
        )
        logger.info(f"Resuming conversation {conversation_id} with user {user_id}")
    else:
        conversation_id = await websocket_manager.connect(user_id, ws)
        await websocket_manager.new_ws_sender(conversation_id, ws).send_message(
            ChatResponse(complete=True, content="Connected!")
        )
        logger.info(f"Starting conversation {conversation_id} with user {user_id}")
    await websocket_manager.wait_for_receive(user_id, conversation_id)
