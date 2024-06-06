from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os

from app.logging_config import logging, setup_logging
from app.openai_client import get_openai_chat_response

# Set up logging
setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI()


class ChatRequest(BaseModel):
    messages: list


@app.get("/")
async def root():
    return {"message": "Hello, World!"}


@app.get("/env")
async def root():
    return {"environ": os.environ}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}


@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        response = get_openai_chat_response(request.messages)
        return {"response": response}
    except Exception as e:
        logger.error(e)
        raise HTTPException(status_code=500, detail=str(e))
