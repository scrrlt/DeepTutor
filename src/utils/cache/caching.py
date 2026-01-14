"""
Caching utilities - TTL cache decorator and semantic caching.
"""

import functools
import time
from typing import Callable, Any


def ttl_cache(seconds: int = 60):
    """
    Simple decorator to cache function results for N seconds.
    Great for caching API keys or Config settings.
    """
    def decorator(func: Callable) -> Callable:
        cache = {}

        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Create a key based on arguments
            key = str(args) + str(kwargs)
            now = time.time()

            if key in cache:
                result, timestamp = cache[key]
                if now - timestamp < seconds:
                    return result

            result = func(*args, **kwargs)
            cache[key] = (result, now)
            return result

        return wrapper
    return decorator
