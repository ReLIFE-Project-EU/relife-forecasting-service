"""Retry utilities for resilient external API calls (e.g., PVGIS)."""

import logging
import time
from functools import wraps
from typing import TypeVar, Callable

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Exceptions that indicate a transient network issue worth retrying.
TRANSIENT_EXCEPTIONS = (ConnectionError, TimeoutError, OSError)

DEFAULT_MAX_RETRIES = 3
DEFAULT_INITIAL_DELAY = 1.0
DEFAULT_BACKOFF_FACTOR = 2.0


def retry_on_transient_error(
    max_retries: int = DEFAULT_MAX_RETRIES,
    initial_delay: float = DEFAULT_INITIAL_DELAY,
    backoff_factor: float = DEFAULT_BACKOFF_FACTOR,
):
    """Decorator that retries a function on transient network errors.

    Uses exponential backoff between attempts.  Only retries on exceptions
    that are subclasses of ``TRANSIENT_EXCEPTIONS``; all other exceptions
    propagate immediately.
    """

    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        @wraps(fn)
        def wrapper(*args, **kwargs) -> T:
            delay = initial_delay
            last_exc: BaseException | None = None
            for attempt in range(1, max_retries + 1):
                try:
                    return fn(*args, **kwargs)
                except TRANSIENT_EXCEPTIONS as exc:
                    last_exc = exc
                    if attempt < max_retries:
                        logger.warning(
                            "Transient error in %s (attempt %d/%d): %s — retrying in %.1fs",
                            fn.__name__,
                            attempt,
                            max_retries,
                            exc,
                            delay,
                        )
                        time.sleep(delay)
                        delay *= backoff_factor
                    else:
                        logger.error(
                            "Transient error in %s (attempt %d/%d): %s — no retries left",
                            fn.__name__,
                            attempt,
                            max_retries,
                            exc,
                        )
            raise last_exc  # type: ignore[misc]

        return wrapper

    return decorator
