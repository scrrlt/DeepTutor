"""LLM telemetry helpers.

Provides basic decorators for telemetry on LLM calls with minimal overhead.
Uses lazy logging to avoid string formatting when log level is disabled.
"""

from collections.abc import Callable
import functools
import inspect
import inspect
import logging
import time
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

# TypeVar for preserving return type through decorator
R = TypeVar("R")


def track_llm_call(provider_name: str) -> Callable:
    """Track LLM calls for telemetry.

    Optimized to minimize overhead on the event loop by:
    - Checking if logging is enabled before string formatting
    - Using lazy %s formatting instead of f-strings
    - Tracking duration to detect slow/hanging calls

    Args:
        provider_name: Name of the provider being called

    Returns:
        Decorator function

    Raises:
        None.

    Raises:
        None.
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Fast fail: avoid string formatting if logging is disabled
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "LLM_START provider=%s func=%s",
                    provider_name,
                    func.__name__,
                )

            start_time = time.perf_counter()
            try:
                result = await func(*args, **kwargs)
                duration_ms = (time.perf_counter() - start_time) * 1000

                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "LLM_SUCCESS provider=%s duration_ms=%.2f",
                        provider_name,
                        duration_ms,
                    )
                return result
            except Exception as e:
                duration_ms = (time.perf_counter() - start_time) * 1000
                logger.warning(
                    "LLM_FAILURE provider=%s duration_ms=%.2f error=%s",
                    provider_name,
                    duration_ms,
                    str(e),
                )
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
