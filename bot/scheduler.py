from apscheduler.schedulers.asyncio import AsyncIOScheduler
import argparse
import asyncio
import signal
import traceback

from bot.controllers.english import english_controller
from bot.graph.idea import idea_graph
from db.utils import db_stats_job
from observability.logging import logging, setup_logging

setup_logging()
logger = logging.getLogger(__name__)


async def main():
    logger.info("Scheduler is starting")
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    scheduler.start()

    await idea_graph.add_default_semantic_categories()
    idea_graph.add_code_block_merge_event_handler(english_controller)
    idea_graph.add_paragraph_merge_event_handler(english_controller)
    idea_graph.add_sentence_merge_event_handler(english_controller)

    while True:
        await asyncio.sleep(10)


def signal_handler(sig, frame):
    logger.info(f"{sig} received. Shutting down...")
    scheduler.shutdown()
    asyncio.get_event_loop().stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Build a graph of ideas.')
    parser.add_argument("--english-controller", help='Enable English controller.',
                        action="store_true", default=False)
    args = parser.parse_args()
    scheduler = AsyncIOScheduler()
    scheduler.add_job(db_stats_job, "interval", minutes=15, name="PostgreSQL stats")

    if args.english_controller:
        scheduler.add_job(english_controller.on_periodic_run, "interval",
                          seconds=english_controller.interval_seconds, name="EntityController")

    try:
        asyncio.run(main())
    except Exception as e:
        logger.error(f"Uncaught in main loop: {traceback.format_exc()}")
        exit(1)
    finally:
        logger.info("Scheduler has stopped")
