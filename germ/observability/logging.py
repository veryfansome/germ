import logging
import logging.config

from germ.settings import germ_settings

console_only_logger_config = {
    "handlers": ["console"],
    "propagate": False,
}


def setup_logging(
        germ_log_level: str = germ_settings.GERM_LOG_LEVEL,
        global_level: str = germ_settings.LOG_LEVEL
):
    logging.config.dictConfig({
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
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
            "germ.services.bot.chat.controller": {
                "level": germ_log_level,
                **console_only_logger_config,
            },
            "httpx": {
                "level": "ERROR",
                **console_only_logger_config,
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
