from apscheduler.schedulers.asyncio import AsyncIOScheduler
from contextlib import asynccontextmanager
from fastapi import Cookie, Depends, FastAPI, File, Form, HTTPException, UploadFile, WebSocket, status
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from opentelemetry import trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.responses import Response
from typing import List
from urllib.parse import urlencode
import aiofiles
import asyncio
import os

from bot.api.models import CookieData
from bot.auth import (MAX_COOKIE_AGE, USER_COOKIE, add_new_user, get_decoded_cookie,
                      get_user_id, get_user_password_hash, sign_cookie, verify_password)
from bot.chat.controller import ChatController
from bot.chat.openai_beta import AssistantHelper
from bot.chat.openai_handlers import ChatRoutingEventHandler, UserProfilingHandler
from bot.db.models import DATABASE_URL, SessionLocal, engine
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

    yield

    # Stopping

    websocket_manager_disconnect_task = asyncio.create_task(websocket_manager.disconnect_all())
    await assistant_helper.no_loose_files()
    await assistant_helper.no_loose_threads()
    await neo4j_driver.shutdown()
    await websocket_manager_disconnect_task
    scheduler.shutdown()


async def require_user(germ_user: str | None = Cookie(None)):
    cookie_data = get_decoded_cookie(germ_user)
    if cookie_data is None:
        # for HTML requests we prefer a redirect instead of JSON 401
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
    return cookie_data


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


@bot.get("/favicon.ico", include_in_schema=False)
async def get_favicon():
    file_path = os.path.join(os.path.dirname(__file__), "static", "assets", "favicon.ico")
    return FileResponse(file_path)


@bot.get("/healthz")
async def get_healthz():
    # Is PostgreSQL usable?
    pg_session = SessionLocal()
    pg_session.close()
    # Is OpenAI usable?
    openai_client = OpenAI()
    openai_client.close()
    return {
        "db_url": DATABASE_URL,
        "environ": os.environ,
        "status": "OK",
    }


@bot.get("/", include_in_schema=False)
async def get_landing(germ_user: str | None = Cookie(None)):
    if get_decoded_cookie(germ_user) is None:
        return RedirectResponse(url="/login", status_code=status.HTTP_303_SEE_OTHER)
    file_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    return FileResponse(
        file_path,
        headers={
            "Cache-Control": "no-store"  # Prevents serving landing from cache after logout
        }
    )


@bot.get("/login", include_in_schema=False)
async def get_login_form(germ_user: str | None = Cookie(None)):
    if get_decoded_cookie(germ_user) is not None:
        return RedirectResponse(url="/", status_code=status.HTTP_303_SEE_OTHER)
    file_path = os.path.join(os.path.dirname(__file__), "static", "login.html")
    return FileResponse(file_path)


@bot.get("/logout", include_in_schema=False)
async def get_logout(return_url: str | None = "/login"):
    if return_url not in {"/login", "/register"}:
        logger.warning(f"Overrode user supplied `return_url` {return_url}, which is not allowed")
        return_url = "/login"
    response = RedirectResponse(url=return_url, status_code=status.HTTP_303_SEE_OTHER)
    response.delete_cookie(USER_COOKIE)
    return response


@bot.get("/metrics")
async def get_metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@bot.get("/register", include_in_schema=False)
async def get_register(germ_user: str | None = Cookie(None)):
    if get_decoded_cookie(germ_user) is not None:
        return RedirectResponse(url="/logout?return_url=/register", status_code=status.HTTP_303_SEE_OTHER)
    file_path = os.path.join(os.path.dirname(__file__), "static", "register.html")
    return FileResponse(file_path)


@bot.get("/user/info", include_in_schema=False)
async def get_user_info(germ_user: str | None = Cookie(None)):
    cookie_data = get_decoded_cookie(germ_user)
    if cookie_data is None:
        return CookieData(user_id=-1, username="unknown")
    return cookie_data


@bot.post("/login", include_in_schema=False)
async def post_login_form(username: str = Form(...), password: str = Form(...)):
    password_hash = await get_user_password_hash(username)
    if password_hash is None:
        error_params = urlencode({"error": "No such user."})
    elif not verify_password(password, password_hash):
        error_params = urlencode({"error": "Invalid password."})
    else:
        user_id = await get_user_id(username)
        response = RedirectResponse("/", status_code=status.HTTP_303_SEE_OTHER)
        response.set_cookie(
            key=USER_COOKIE,
            value=sign_cookie(CookieData(user_id=user_id, username=username).model_dump_json()),
            httponly=True,
            samesite="lax",
            max_age=MAX_COOKIE_AGE,
        )
        return response
    return RedirectResponse(f"/login?{error_params}", status_code=status.HTTP_303_SEE_OTHER)



@bot.post("/register", include_in_schema=False)
async def post_register(username: str = Form(...), password: str = Form(...), verify: str = Form(...)):
    if (await get_user_password_hash(username)) is not None:
        error_params = urlencode({"error": "Username already exists."})
    elif password != verify:
        error_params = urlencode({"error": "'Password' and 'Verify password' did not match."})
    else:
        user_id = await add_new_user(username, password)
        if user_id is None:
            error_params = urlencode({"error": "Failed to create new user."})
        else:
            await control_plane.add_chat_user(user_id)
            response = RedirectResponse("/", status_code=status.HTTP_303_SEE_OTHER)
            response.set_cookie(
                key=USER_COOKIE,
                value=sign_cookie(CookieData(user_id=user_id, username=username).model_dump_json()),
                httponly=True,
                samesite="lax",
                max_age=MAX_COOKIE_AGE,
            )
            return response
    return RedirectResponse(f"/register?{error_params}", status_code=status.HTTP_303_SEE_OTHER)


@bot.post("/upload")
async def post_upload(files: List[UploadFile] = File(...), user_id: str = Depends(require_user)):
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
    cookie_data = get_decoded_cookie(ws.cookies.get(USER_COOKIE))
    if cookie_data is None:
        # Fail with 4401 so browser JS gets onclose
        await ws.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    chat_session_id = await websocket_manager.connect(cookie_data.user_id, ws)
    await control_plane.link_chat_user_to_chat_session(cookie_data.user_id, chat_session_id)
    logger.info(f"starting session {chat_session_id} with user {cookie_data.user_id}")
    await websocket_manager.conduct_chat_session(cookie_data.user_id, chat_session_id)
