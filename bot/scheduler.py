import argparse
import asyncio
import signal
import traceback
from apscheduler.schedulers.asyncio import AsyncIOScheduler

import bot.lang.dependencies  # For model downloads
from bot.graph.idea import get_idea_graph
from bot.think import core_identity
from bot.think import idea_distillation
from db.utils import db_stats_job
from observability.logging import logging, setup_logging

setup_logging()
logger = logging.getLogger(__name__)

idea_graph = get_idea_graph(__name__)


async def main():
    logger.info("Scheduler is starting")
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    scheduler.start()

    await core_identity.main()
    while True:
        await asyncio.sleep(10)


def signal_handler(sig, frame):
    logger.info(f"{sig} received. Shutting down...")
    scheduler.shutdown()
    asyncio.get_event_loop().stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Build a graph of ideas.')
    parser.add_argument("--core-identity", help='Enable core identity reinforcement runs.',
                        action="store_true", default=False)
    parser.add_argument("--idea-distillation", help='Enable idea distillation runs.',
                        action="store_true", default=False)
    args = parser.parse_args()
    scheduler = AsyncIOScheduler()
    scheduler.add_job(db_stats_job, "interval", minutes=15, name="PostgreSQL stats")

    if args.core_identity:
        scheduler.add_job(core_identity.main, "interval", minutes=15, name="Core identity")
    if args.idea_distillation:
        scheduler.add_job(idea_distillation.main, "interval", minutes=5, name="Idea distillation")

    try:
        asyncio.run(main())
    except Exception as e:
        logger.error(f"Uncaught in main loop: {traceback.format_exc()}")
        exit(1)
    finally:
        logger.info("Scheduler has stopped")
