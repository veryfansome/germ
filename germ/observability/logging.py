import logging
import logging.config

from germ.settings import germ_settings

console_only_logger_config = {
    "handlers": ["console"],
    "propagate": False,
}


def setup_logging(global_level: str = germ_settings.LOG_LEVEL,
                  log_dir: str = germ_settings.LOG_DIR,
                  message_log_filename: str = germ_settings.MESSAGE_LOG_FILENAME):
    logging.config.dictConfig({
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            },
            "message_only": {
                "format": "%(message)s",
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "default",
            },
            "message": {
                "filename": f"{log_dir}/{message_log_filename}",
                "class": "logging.handlers.TimedRotatingFileHandler",
                "formatter": "message_only",
                # Daily rotation with 7 + 1 days of history
                "when": "D",
                "interval": 1,
                "encoding": "utf8",
            },
        },
        "loggers": {
            "": {
                "level": global_level,
                "handlers": ["console"],
            },
            "httpx": {
                "level": "ERROR",
                **console_only_logger_config,
            },
            "message": {
                "level": "INFO",
                "handlers": ["message"],
                "propagate": False,
            },
            "neo4j.notifications": {
                "level": "ERROR",
                **console_only_logger_config,
            },
            "sqlalchemy": {
                "level": "ERROR",
                **console_only_logger_config,
            },
            "uvicorn.error": {
                "level": "ERROR",
                **console_only_logger_config,
            },
            "uvicorn.access": {
                "level": "INFO",
                **console_only_logger_config,
            },
        }}
    )
