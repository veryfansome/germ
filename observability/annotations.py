from functools import wraps
from prometheus_client import Histogram
import logging
import time

logger = logging.getLogger(__name__)

FUNC_EXEC_SECONDS = Histogram(f'func_exec_seconds', 'Time spent executing function', ['func_name'])


def measure_exec_seconds(use_logging: bool = False, use_prometheus: bool = False):

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            time_started = time.time()
            try:
                return func(*args, **kwargs)
            finally:
                time_elapsed = time.time() - time_started
                if use_prometheus:
                    FUNC_EXEC_SECONDS.labels(func_name=func.__name__).observe(time_elapsed)
                if use_logging:
                    logger.info(f"{func.__name__} exec took {time_elapsed}s")
        return wrapper
    return decorator
