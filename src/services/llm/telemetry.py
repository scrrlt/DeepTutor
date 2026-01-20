# -*- coding: utf-8 -*-
"""
LLM Telemetry
=============

Basic telemetry tracking for LLM calls.
"""

from collections.abc import AsyncGenerator, Callable
import functools
import inspect
import logging
import time
from typing import Any, TypeVar

from src.logging.stats import llm_stats

logger = logging.getLogger(__name__)

# TypeVar for preserving decorated function type information
F = TypeVar("F", bound=Callable[..., Any])


def track_llm_call(provider_name: str) -> Callable[[F], F]:
    """
    Decorator to track LLM calls for telemetry.

    Args:
        provider_name: Name of the provider being called

    Returns:
        Decorator function

    Raises:
        None.
    """

    def decorator(func: Any) -> Any:
        if inspect.isasyncgenfunction(func):

            @functools.wraps(func)
            async def generator_wrapper(*args, **kwargs):
                start_time = time.perf_counter()
                logger.debug("LLM call to %s: %s", provider_name, func.__name__)
                # Instantiation errors are rare; iteration errors are handled in _wrap_stream.
                stream = func(*args, **kwargs)

                async for chunk in _wrap_stream(provider_name, stream, start_time):
                    yield chunk

            return generator_wrapper

        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
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

            duration = time.perf_counter() - start_time
            llm_stats.record_latency(provider_name, duration)

            usage = getattr(result, "usage", None)
            if isinstance(usage, dict) and usage:
                total_tokens = usage.get("total_tokens")
                if total_tokens is None:
                    total_tokens = usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)
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
    stream: AsyncGenerator[Any, None],
    start_time: float,
) -> AsyncGenerator[Any, None]:
    """
    Wrap an async generator to record streaming telemetry metrics.

    Args:
        provider_name: Provider name for metrics.
        stream: Async generator returned by the provider.
        start_time: Perf counter start time for the request.

    Yields:
        Streamed chunks from the provider.

    Raises:
        Exception: Propagates stream iteration errors.
    """
    first_chunk_time: float | None = None
    completed = False
    try:
        async for chunk in stream:
            if first_chunk_time is None and _has_stream_payload(chunk):
                first_chunk_time = time.perf_counter()
                llm_stats.record_ttft(provider_name, first_chunk_time - start_time)
            yield chunk
        completed = True
    except Exception as exc:
        llm_stats.record_error(provider_name, type(exc).__name__)
        raise
    finally:
        duration = time.perf_counter() - start_time
        llm_stats.record_latency(provider_name, duration)
        if completed:
            logger.debug("LLM stream to %s completed successfully", provider_name)


def _has_stream_payload(chunk: object) -> bool:
    """
    Determine whether a stream chunk contains a payload worth timing.

    Args:
        chunk: Stream chunk emitted by a provider.

    Returns:
        True when the chunk includes meaningful text content.

    Raises:
        None.
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
