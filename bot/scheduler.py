from apscheduler.schedulers.asyncio import AsyncIOScheduler
import argparse
import asyncio
import signal
import traceback

from bot.db.neo4j import AsyncNeo4jDriver
from bot.db.utils import db_stats_job
from bot.graph.control_plane import ControlPlane
from bot.lang.controllers.english import EnglishController
from observability.logging import logging, setup_logging

setup_logging()
logger = logging.getLogger(__name__)


class Scheduler:
    def __init__(self, args, asyncio_scheduler):
        self.args = args
        self.asyncio_scheduler = asyncio_scheduler

        self.neo4j_driver = AsyncNeo4jDriver()
        self.control_plane = ControlPlane(self.neo4j_driver)

    async def run(self):
        loop = asyncio.get_event_loop()
        loop.add_signal_handler(signal.SIGINT, self.shutdown)
        loop.add_signal_handler(signal.SIGTERM, self.shutdown)

        logger.info("Scheduler starting")
        await self.control_plane.initialize()

        self.asyncio_scheduler.add_job(db_stats_job, "interval", minutes=15, name="PostgreSQL stats")

        if self.args.english_controller:
            english_controller = EnglishController(self.control_plane)
            self.asyncio_scheduler.add_job(english_controller.on_periodic_run, "interval",
                                           seconds=english_controller.interval_seconds, name="EntityController")
            self.control_plane.add_code_block_merge_event_handler(english_controller)
            self.control_plane.add_paragraph_merge_event_handler(english_controller)
            self.control_plane.add_sentence_merge_event_handler(english_controller)

        self.asyncio_scheduler.start()
        while True:
            await asyncio.sleep(10)

    def shutdown(self):
        asyncio.run(self.neo4j_driver.shutdown())
        self.asyncio_scheduler.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Build a graph of ideas.')
    parser.add_argument("--english-controller", help='Enable English controller.',
                        action="store_true", default=False)
    exit_code = 0
    try:
        scheduler = Scheduler(parser.parse_args(), AsyncIOScheduler())
        asyncio.run(scheduler.run())
    except Exception as e:
        logger.error(f"Uncaught in main loop: {traceback.format_exc()}")
        exit_code = 1
    finally:
        logger.info("Scheduler has stopped")
    exit(exit_code)
