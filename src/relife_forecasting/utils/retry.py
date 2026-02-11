"""Retry utilities for resilient external API calls (e.g., PVGIS)."""

import asyncio
import inspect
import logging
import time
from functools import wraps
from typing import TypeVar, Callable

try:
    # `requests` is used by the calling code; include its base exception when available.
    from requests.exceptions import RequestException as _RequestsRequestException
except Exception:  # ImportError or if `requests` is not installed
    _RequestsRequestException = None  # type: ignore[assignment]

try:
    # `urllib3` underpins `requests`; include its HTTPError when available.
    from urllib3.exceptions import HTTPError as _Urllib3HTTPError
except Exception:  # ImportError or if `urllib3` is not installed
    _Urllib3HTTPError = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Exceptions that indicate a transient network issue worth retrying.
_transient_exceptions: list[type[BaseException]] = [
    ConnectionError,
    TimeoutError,
    OSError,
]

if _RequestsRequestException is not None:
    _transient_exceptions.append(_RequestsRequestException)

if _Urllib3HTTPError is not None:
    _transient_exceptions.append(_Urllib3HTTPError)

TRANSIENT_EXCEPTIONS = tuple(_transient_exceptions)
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
    
    Supports both sync and async functions; will automatically detect coroutines
    and use asyncio.sleep instead of time.sleep to avoid blocking the event loop.
    """
    if max_retries < 1:
        raise ValueError(f"max_retries must be >= 1, got {max_retries}")
    if initial_delay < 0:
        raise ValueError(f"initial_delay must be >= 0, got {initial_delay}")
    if backoff_factor <= 0:
        raise ValueError(f"backoff_factor must be > 0, got {backoff_factor}")

    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        is_coroutine = inspect.iscoroutinefunction(fn)
        
        @wraps(fn)
        async def async_wrapper(*args, **kwargs) -> T:
            delay = initial_delay
            last_exc: BaseException | None = None
            for attempt in range(1, max_retries + 1):
                try:
                    return await fn(*args, **kwargs)
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
                        await asyncio.sleep(delay)
                        delay *= backoff_factor
                    else:
                        logger.error(
                            "Transient error in %s (attempt %d/%d): %s — no retries left",
                            fn.__name__,
                            attempt,
                            max_retries,
                            exc,
                        )
            if last_exc is not None:
                raise last_exc
            raise RuntimeError(
                "retry_on_transient_error exhausted retries without capturing an exception"
            )

        @wraps(fn)
        def sync_wrapper(*args, **kwargs) -> T:
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
            if last_exc is not None:
                raise last_exc
            raise RuntimeError(
                "retry_on_transient_error exhausted retries without capturing an exception"
            )

        return async_wrapper if is_coroutine else sync_wrapper  # type: ignore[return-value]

    return decorator
