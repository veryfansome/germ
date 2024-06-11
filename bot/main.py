from fastapi import FastAPI, HTTPException, Path
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from openai import OpenAI
from pydantic import BaseModel
from starlette.requests import Request
import hashlib
import os
import subprocess

from bot.db_chat_history import (DATABASE_URL as CHAT_HISTORY_DATABASE_URL,
                                 SessionLocal as ChatHistorySessionLocal,
                                 MessageBookmark, MessageThumbsDown)
from bot.logging_config import logging, setup_logging, traceback
from bot.v1 import ChatBookmark, ChatRequest, LinkedMessageIds, OpenAIChatBot


setup_logging()
logger = logging.getLogger(__name__)

bot = FastAPI()
bot.mount("/static", StaticFiles(directory="bot/static"), name="static")
templates = Jinja2Templates(directory="bot/templates")

chat_bot = OpenAIChatBot()


class SqlRequest(BaseModel):
    sql: str


@bot.get("/", include_in_schema=False)
async def get(request: Request):
    return templates.TemplateResponse(request, "index.html")


@bot.post("/chat")
async def chat(request: ChatRequest):
    try:
        response = chat_bot.chat(request.messages)
        return response
    except Exception as e:
        logger.error("%s: trace: %s", e, traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@bot.post("/chat/bookmark")
async def chat(request: ChatBookmark):
    with ChatHistorySessionLocal() as session:
        result = session.query(MessageBookmark).where(
            MessageBookmark.message_received_id == request.message_received_id,
            MessageBookmark.message_sent_id == request.message_sent_id
        ).one_or_none()
        if result:
            return {"bookmark_id": result.id}
    try:
        bookmark = MessageBookmark(
            message_received_id=request.message_received_id,
            message_summary=chat_bot.summarize_text(request.message_sent_content).encode("utf-8"),
            message_sent_id=request.message_sent_id)
        with ChatHistorySessionLocal() as session:
            session.add(bookmark)
            session.commit()
            session.refresh(bookmark)
        return {"bookmark_id": bookmark.id}
    except Exception as e:
        logger.error("%s: trace: %s", e, traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@bot.post("/chat/thumbs-down")
async def chat(request: LinkedMessageIds):
    try:
        thumbs_down = MessageThumbsDown(
            message_received_id=request.message_received_id,
            message_sent_id=request.message_sent_id)
        with ChatHistorySessionLocal() as session:
            session.add(thumbs_down)
            session.commit()
            session.refresh(thumbs_down)
        return {"thumbs_down_id": thumbs_down.id}
    except Exception as e:
        logger.error("%s: trace: %s", e, traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@bot.get("/env", include_in_schema=False)
async def root():
    return {"environ": os.environ}


@bot.get("/favicon.ico", include_in_schema=False)
def favicon():
    favicon_path = os.path.join(os.path.dirname(__file__), 'static', 'favicon.ico')
    return FileResponse(favicon_path)


@bot.get("/healthz")
async def healthz():
    # Is PostgreSQL usable?
    session = ChatHistorySessionLocal()
    session.close()
    # Is OpenAI usable?
    openai_client = OpenAI()
    openai_client.close()
    return {
        "status": "OK",
        "chat_history_db_url": CHAT_HISTORY_DATABASE_URL
    }


@bot.get("/logo.webp", include_in_schema=False)
def favicon():
    favicon_path = os.path.join(os.path.dirname(__file__), 'static', 'logo.webp')
    return FileResponse(favicon_path)


@bot.get("/postgres.html", include_in_schema=False)
async def get(request: Request):
    return templates.TemplateResponse(request, "postgres.html")


@bot.post("/postgres/{db}/query")
async def postgres_query(request: SqlRequest,
                         db: str = Path(..., title="Postgres DB")):
    enabled_dbs = {
        "chat_history": CHAT_HISTORY_DATABASE_URL
    }
    if db not in enabled_dbs:
        raise HTTPException(status_code=400, detail="Not supported")

    md5 = hashlib.md5()
    md5.update(request.sql.encode())
    query_file = f"/tmp/{db}_query_{md5.hexdigest()}.sql"
    with open(query_file, "w") as f:
        f.write(request.sql)
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


@bot.get("/ui.css", include_in_schema=False)
def favicon():
    favicon_path = os.path.join(os.path.dirname(__file__), 'static', 'ui.css')
    return FileResponse(favicon_path)
