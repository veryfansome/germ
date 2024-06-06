from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from openai import OpenAI
from starlette.requests import Request
import os

from bot.db_chat_history import DATABASE_URL as CHAT_HISTORY_DATABASE_URL, SessionLocal as ChatHistorySessionLocal
from bot.logging_config import logging, setup_logging
from bot.v1 import ChatRequest, OpenAIChatBot


setup_logging()
logger = logging.getLogger(__name__)

bot = FastAPI()
bot.mount("/static", StaticFiles(directory="bot/static"), name="static")
templates = Jinja2Templates(directory="bot/templates")

chat_bot = OpenAIChatBot()


@bot.get("/")
async def get(request: Request):
    return templates.TemplateResponse(request, "index.html")


@bot.get("/favicon.ico", include_in_schema=False)
def favicon():
    favicon_path = os.path.join(os.path.dirname(__file__), 'static', 'favicon.ico')
    return FileResponse(favicon_path)


@bot.post("/chat")
async def chat(request: ChatRequest):
    try:
        response = chat_bot.chat(request.messages)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@bot.get("/env")
async def root():
    return {"environ": os.environ}


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
