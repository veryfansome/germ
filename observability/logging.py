import logging
import logging.config
from settings import germ_settings

LOGGING_CONFIG = {
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
            "level": germ_settings.LOG_LEVEL,
            "handlers": ["console"],
        },
        "uvicorn.error": {
            "level": germ_settings.LOG_LEVEL,
            "handlers": ["console"],
            "propagate": False,
        },
        "uvicorn.access": {
            "level": germ_settings.LOG_LEVEL,
            "handlers": ["console"],
            "propagate": False,
        },
    },
}


def setup_logging():
    logging.config.dictConfig(LOGGING_CONFIG)
