import argparse
import asyncio
import signal
import traceback
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from bot.graph.idea import idea_graph
from bot.graph import inconsistency_finder
from db.utils import db_stats_job
from observability.logging import logging, setup_logging

setup_logging()
logger = logging.getLogger(__name__)


async def main():
    logger.info("Scheduler is starting")
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    scheduler.start()

    while True:
        await asyncio.sleep(10)


def signal_handler(sig, frame):
    logger.info(f"{sig} received. Shutting down...")
    scheduler.shutdown()
    asyncio.get_event_loop().stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Build a graph of ideas.')
    parser.add_argument("--inconsistency-finder", help='Enable idea distillation runs.',
                        action="store_true", default=False)
    args = parser.parse_args()
    scheduler = AsyncIOScheduler()
    scheduler.add_job(db_stats_job, "interval", minutes=15, name="PostgreSQL stats")

    if args.inconsistency_finder:
        #scheduler.add_job(inconsistency_finder.main, "interval", minutes=1, name="Inconsistency finder")
        pass

    try:
        asyncio.run(main())
    except Exception as e:
        logger.error(f"Uncaught in main loop: {traceback.format_exc()}")
        exit(1)
    finally:
        logger.info("Scheduler has stopped")
