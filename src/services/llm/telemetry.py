# -*- coding: utf-8 -*-
"""
LLM Telemetry
=============

Basic telemetry tracking for LLM calls.
"""

import functools
import inspect
import logging
import time
from typing import AsyncGenerator, Awaitable, Callable, ParamSpec, TypeVar

from src.logging.stats import llm_stats

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
            start_time = time.perf_counter()
            logger.debug("LLM call to %s: %s", provider_name, func.__name__)
            try:
                result = func(*args, **kwargs)
                if inspect.isawaitable(result):
                    result = await result
            except Exception as e:
                llm_stats.record_error(provider_name, type(e).__name__)
                logger.warning("LLM call to %s failed: %s", provider_name, e)
                raise

            if inspect.isasyncgen(result):
                return _wrap_stream(provider_name, result, start_time)

            duration = time.perf_counter() - start_time
            llm_stats.record_latency(provider_name, duration)

            usage = getattr(result, "usage", None)
            if isinstance(usage, dict) and usage:
                total_tokens = usage.get("total_tokens")
                if total_tokens is None:
                    total_tokens = usage.get("prompt_tokens", 0) + usage.get(
                        "completion_tokens", 0
                    )
                llm_stats.record_usage(
                    provider=provider_name,
                    model=getattr(result, "model", "unknown"),
                    tokens=int(total_tokens or 0),
                    cost=float(getattr(result, "cost_estimate", 0.0) or 0.0),
                )

            logger.debug("LLM call to %s completed successfully", provider_name)
            return result

        return wrapper

    return decorator


async def _wrap_stream(
    provider_name: str,
    stream: AsyncGenerator[T, None],
    start_time: float,
) -> AsyncGenerator[T, None]:
    """
    Wrap an async generator to record streaming telemetry metrics.

    Args:
        provider_name: Provider name for metrics.
        stream: Async generator returned by the provider.
        start_time: Perf counter start time for the request.

    Yields:
        Streamed chunks from the provider.
    """
    first_chunk_time: float | None = None
    try:
        async for chunk in stream:
            if first_chunk_time is None and _has_stream_payload(chunk):
                first_chunk_time = time.perf_counter()
                llm_stats.record_ttft(
                    provider_name, first_chunk_time - start_time
                )
            yield chunk
    except Exception as exc:
        llm_stats.record_error(provider_name, type(exc).__name__)
        raise
    finally:
        duration = time.perf_counter() - start_time
        llm_stats.record_latency(provider_name, duration)


def _has_stream_payload(chunk: object) -> bool:
    """
    Determine whether a stream chunk contains a payload worth timing.

    Args:
        chunk: Stream chunk emitted by a provider.

    Returns:
        True when the chunk includes meaningful text content.
    """
    if isinstance(chunk, str):
        return bool(chunk)

    delta = getattr(chunk, "delta", None)
    if isinstance(delta, str) and delta:
        return True

    content = getattr(chunk, "content", None)
    if isinstance(content, str) and content:
        return True

    return False
