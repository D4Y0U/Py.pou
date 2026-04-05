"""Utility decorators and helpers for aipou."""

import hashlib
import json
import logging
import time
from functools import wraps
from typing import Any, Callable

logger = logging.getLogger(__name__)


def cache_response(func: Callable) -> Callable:
    """
    Decorator: caches the return value of a function based on its keyword arguments.
    The cache lives in memory for the lifetime of the decorated function object.
    """
    cache: dict[str, Any] = {}

    @wraps(func)
    def wrapper(*args, **kwargs):
        key = hashlib.md5(
            json.dumps(kwargs, sort_keys=True, default=str).encode()
        ).hexdigest()
        if key in cache:
            logger.debug("cache_response: hit for %s", func.__name__)
            return cache[key]
        result = func(*args, **kwargs)
        cache[key] = result
        return result

    wrapper.cache_clear = cache.clear  # type: ignore[attr-defined]
    return wrapper


def retry_on_failure(max_retries: int = 3, delay: float = 1.0, exceptions=(Exception,)):
    """
    Decorator: retry a function up to `max_retries` times on `exceptions`,
    with exponential backoff starting at `delay` seconds.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exc: Exception = RuntimeError("No attempts made")
            for attempt in range(1, max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:  # type: ignore[misc]
                    last_exc = exc
                    wait = delay * (2 ** (attempt - 1))
                    logger.warning(
                        "retry_on_failure: %s échec (tentative %d/%d). Pause %.1fs. Erreur: %s",
                        func.__name__, attempt, max_retries, wait, exc,
                    )
                    time.sleep(wait)
            raise last_exc
        return wrapper
    return decorator


def log_call(func: Callable) -> Callable:
    """Decorator: logs the function name, kwargs, and elapsed time on each call."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.info("→ %s appelé avec %s", func.__name__, kwargs)
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        logger.info("← %s terminé en %.3fs", func.__name__, elapsed)
        return result
    return wrapper