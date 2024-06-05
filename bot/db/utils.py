from prometheus_client import Gauge
from starlette.concurrency import run_in_threadpool

from bot.db.models import (SessionLocal,
                           ChatSession, ChatRequestReceived, ChatResponseSent)
from observability.logging import logging

logger = logging.getLogger(__name__)

METRIC_CHAT_SESSION_ROW_COUNT = Gauge(
    "chat_session_row_count", "Number of rows in the chat_session table")

METRIC_CHAT_REQUEST_RECEIVED_ROW_COUNT = Gauge(
    "chat_request_received_row_count", "Number of rows in the chat_request_received table")

METRIC_CHAT_RESPONSE_SENT_ROW_COUNT = Gauge(
    "chat_response_sent_row_count", "Number of rows in the chat_response_sent table")


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
            F"chat_response_sent_count={chat_response_sent_count}")))


async def db_stats_job():
    await run_in_threadpool(db_stats)
