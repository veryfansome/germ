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

from bot.auth import MAX_COOKIE_AGE, SESSION_COOKIE_NAME, AuthHelper, verify_password
from bot.chat.controller import ChatController
from bot.chat.openai_beta import AssistantHelper
from bot.chat.openai_handlers import ChatRoutingEventHandler, UserProfilingHandler
from bot.db.pg import DATABASE_URL, TableHelper
from bot.db.neo4j import AsyncNeo4jDriver
from bot.graph.control_plane import ControlPlane
from bot.websocket import WebSocketConnectionManager
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
# Databases

neo4j_driver = AsyncNeo4jDriver()
pg_engine = create_async_pg_engine(f"postgresql+asyncpg://{DATABASE_URL}", echo=False)
pg_session_maker = async_pg_session_maker(
    bind=pg_engine,
    class_=AsyncSession,
    expire_on_commit=False
)
pg_table_helper = TableHelper(create_pg_engine(f"postgresql+psycopg2://{DATABASE_URL}", echo=False))
redis_client = Redis.from_url(
    f"redis://{germ_settings.REDIS_HOST}:{germ_settings.REDIS_PORT}",
    decode_responses=False
)

##
# App

auth_helper = AuthHelper(pg_table_helper.chat_user_table, pg_session_maker)
control_plane = ControlPlane(neo4j_driver)
websocket_manager = WebSocketConnectionManager(
    pg_table_helper.chat_message_table,
    pg_table_helper.chat_session_table,
    control_plane,
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
        neo4j_driver.shutdown(),
        pg_engine.dispose(),
        pg_table_helper.shutdown(),
        redis_client.close(),
    ])

    logger.info("Stopped")


bot = FastAPI(
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
bot.mount("/static", StaticFiles(directory="bot/static"), name="static")
if os.path.exists("bot/static/tests"):
    bot.mount("/tests", StaticFiles(directory="bot/static/tests"), name="tests")
    bot.mount("/tests/cov", StaticFiles(directory="bot/static/tests/cov"), name="tests_cov")


##
# Enabled instrumentation
FastAPIInstrumentor.instrument_app(bot)
RedisInstrumentor().instrument(client=redis_client)
SQLAlchemyInstrumentor().instrument(engine=pg_engine.sync_engine)  # Async is a facade backed by a sync engine


##
# Endpoints


@bot.get("/favicon.ico", include_in_schema=False)
async def get_favicon():
    file_path = os.path.join(os.path.dirname(__file__), "static", "assets", "favicon.ico")
    return FileResponse(file_path)


@bot.get("/healthz")
async def get_healthz():
    return {
        "environ": os.environ,
        "status": "OK",
    }


@bot.get("/", include_in_schema=False)
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


@bot.get("/login", include_in_schema=False)
async def get_login_form(request:  Request):
    await load_session(request)
    session = request.session
    if session:
        return RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)
    file_path = os.path.join(os.path.dirname(__file__), "static", "login.html")
    return FileResponse(file_path)


@bot.get("/logout", include_in_schema=False)
async def get_logout(request: Request):
    await load_session(request)
    request.session.clear()
    return RedirectResponse(url="/login", status_code=status.HTTP_303_SEE_OTHER)


@bot.get("/metrics")
async def get_metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@bot.get("/register", include_in_schema=False)
async def get_register(request:  Request):
    await load_session(request)
    session = request.session
    if session:
        return RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)
    file_path = os.path.join(os.path.dirname(__file__), "static", "register.html")
    return FileResponse(file_path)


@bot.get("/user/info", include_in_schema=False)
async def get_user_info(request:  Request):
    await load_session(request)
    session = request.session
    if not session:
        return RedirectResponse(f"/login", status_code=status.HTTP_303_SEE_OTHER)
    return session


@bot.post("/login", include_in_schema=False)
async def post_login_form(request: Request, username: str = Form(...), password: str = Form(...)):
    load_session_task = asyncio.create_task(load_session(request))
    password_hash = await auth_helper.get_user_password_hash(username)
    if password_hash is None:
        error_params = urlencode({"error": "No such user."})
    elif not verify_password(password, password_hash):
        error_params = urlencode({"error": "Invalid password."})
    else:
        user_id = await auth_helper.get_user_id(username)
        await load_session_task
        request.session.clear()
        request.session.update({"user_id": user_id, "username": username})
        return RedirectResponse("/", status_code=status.HTTP_303_SEE_OTHER)
    return RedirectResponse(f"/login?{error_params}", status_code=status.HTTP_303_SEE_OTHER)


@bot.post("/register", include_in_schema=False)
async def post_register(request:  Request, username: str = Form(...), password: str = Form(...), verify: str = Form(...)):
    load_session_task = asyncio.create_task(load_session(request))
    if (await auth_helper.get_user_password_hash(username)) is not None:
        error_params = urlencode({"error": "Username already exists."})
    elif password != verify:
        error_params = urlencode({"error": "'Password' and 'Verify password' did not match."})
    else:
        user_id, dt_created = await auth_helper.add_new_user(username, password)
        if user_id is None:
            error_params = urlencode({"error": "Failed to create new user."})
        else:
            await control_plane.add_chat_user(user_id, dt_created)
            await load_session_task
            request.session.clear()
            request.session.update({"user_id": user_id, "username": username})
            return RedirectResponse("/", status_code=status.HTTP_303_SEE_OTHER)
    return RedirectResponse(f"/register?{error_params}", status_code=status.HTTP_303_SEE_OTHER)


@bot.post("/upload")
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


@bot.websocket("/chat")
async def websocket_chat(ws: WebSocket):
    await ws.accept()
    await load_session(ws)
    session = ws.scope.get("session")
    if not session or "user_id" not in session:
        await ws.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    user_id = session["user_id"]
    session_id = await websocket_manager.connect(user_id, ws)
    logger.info(f"starting session {session_id} with user {user_id}")
    await websocket_manager.wait_for_receive(user_id, session_id)
