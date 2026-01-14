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
        retries: Number of retry attempts
        base_delay: Base delay between retries
        jitter: Random jitter to add to delay
        retry_on: Exception types to retry on

    Returns:
        Function result

    Raises:
        RetryError: If all retries are exhausted
    """
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
