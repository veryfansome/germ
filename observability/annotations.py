from functools import wraps
from prometheus_client import Histogram
import logging
import time

logger = logging.getLogger(__name__)

FUNC_EXEC_SECONDS = Histogram(f'func_exec_seconds', 'Time spent executing function', ['func_name'])


def get_func_name(func):
    module = getattr(func, '__module__', '<unknown module>')
    # Check if 'func' is a bound method (i.e., has a __self__ attribute and it is not None)
    if hasattr(func, '__self__') and func.__self__ is not None:
        # Optionally, get the class name from the instance:
        cls_name = func.__self__.__class__.__name__
        return f"{module}.{cls_name}.{func.__name__}"
    else:
        # For unbound functions, use __qualname__ if available
        qualname = getattr(func, '__qualname__', func.__name__)
        return f"{module}.{qualname}"


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
                    FUNC_EXEC_SECONDS.labels(
                        func_name=get_func_name(func)).observe(time_elapsed)
                if use_logging:
                    logger.info(f"{get_func_name(func)} exec took {time_elapsed}s")
        return wrapper
    return decorator
