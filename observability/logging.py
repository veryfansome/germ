import logging
import logging.config
from settings import germ_settings


def setup_logging(global_level: str = germ_settings.LOG_LEVEL):
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                },
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "default",
                },
            },
            "loggers": {
                "": {
                    "level": global_level,
                    "handlers": ["console"],
                },
                "httpx": {
                    "level": "ERROR",
                    "handlers": ["console"],
                },
                "neo4j.notifications": {
                    "level": "ERROR",
                    "handlers": ["console"],
                },
                "sqlalchemy.engine.Engine": {
                    "level": "ERROR",
                    "handlers": ["console"],
                },
                "uvicorn.error": {
                    "level": "ERROR",
                    "handlers": ["console"],
                    "propagate": False,
                },
                "uvicorn.access": {
                    "level": "INFO",
                    "handlers": ["console"],
                    "propagate": False,
                },
            },
        }
    )
