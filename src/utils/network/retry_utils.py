"""
Retry Utilities - Exponential backoff retry decorator.
"""

import random
import time
from typing import Any, Callable, Tuple, Type


class RetryError(Exception):
    """Raised when all retry attempts are exhausted."""
    pass


def retry(
    fn: Callable,
    *,
    retries: int = 3,
    base_delay: float = 0.5,
    jitter: float = 0.2,
    retry_on: Tuple[Type[Exception], ...] = (Exception,)
) -> Any:
    """
    Retry a function with exponential backoff.

    Args:
        fn: Function to retry
        retries: Number of retry attempts (must be >= 0)
        base_delay: Base delay between retries (must be >= 0)
        jitter: Random jitter to add to delay (must be >= 0)
        retry_on: Exception types to retry on

    Returns:
        Function result

    Raises:
        RetryError: If all retries are exhausted
        ValueError: If retries, base_delay, or jitter are negative
    """
    if retries < 0:
        raise ValueError("retries must be non-negative")
    if base_delay < 0:
        raise ValueError("base_delay must be non-negative")
    if jitter < 0:
        raise ValueError("jitter must be non-negative")

    for attempt in range(retries):
        try:
            return fn()
        except retry_on as exc:
            if attempt == retries - 1:
                raise RetryError(str(exc)) from exc
            sleep_for = base_delay * (2 ** attempt) + random.uniform(0, jitter)
            time.sleep(sleep_for)
    # This should never be reached, but added for linter satisfaction
    raise RetryError("Unexpected error in retry logic")
