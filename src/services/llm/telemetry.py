# -*- coding: utf-8 -*-
"""
LLM Telemetry
=============

Basic telemetry tracking for LLM calls.
"""

import functools
import logging
from typing import Awaitable, Callable, ParamSpec, TypeVar

logger = logging.getLogger(__name__)

P = ParamSpec("P")
T = TypeVar("T")


def track_llm_call(
    provider_name: str,
) -> Callable[[Callable[P, Awaitable[T]]], Callable[P, Awaitable[T]]]:
    """
    Decorator to track LLM calls for telemetry.

    Args:
        provider_name: Name of the provider being called

    Returns:
        Decorator function
    """

    def decorator(
        func: Callable[P, Awaitable[T]],
    ) -> Callable[P, Awaitable[T]]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            logger.debug(f"LLM call to {provider_name}: {func.__name__}")
            try:
                result = await func(*args, **kwargs)
                logger.debug(
                    f"LLM call to {provider_name} completed successfully"
                )
                return result
            except Exception as e:
                logger.warning(f"LLM call to {provider_name} failed: {e}")
                raise

        return wrapper

    return decorator
