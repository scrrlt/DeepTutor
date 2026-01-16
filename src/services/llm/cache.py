"""Redis-backed cache helpers for LLM completions."""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
from typing import Any, Dict, List, Optional

from redis.asyncio import Redis

from src.logging import get_logger

logger = get_logger("LLMCache")

_CACHE_CLIENT: Optional[Redis] = None
_CACHE_LOCK: Optional[asyncio.Lock] = None
CACHE_NAMESPACE = os.getenv("LLM_CACHE_NAMESPACE", "llm_cache")


def _get_default_cache_ttl() -> int:
    """Safely parse cache TTL from environment with clear errors."""
    raw_ttl = os.getenv("LLM_CACHE_TTL", "3600")
    try:
        return int(raw_ttl)
    except ValueError as exc:
        raise ValueError(
            f"LLM_CACHE_TTL must be an integer, got {raw_ttl!r}."
        ) from exc


DEFAULT_CACHE_TTL = _get_default_cache_ttl()


async def get_cache_client() -> Optional[Redis]:
    """Get or create a Redis client for caching.

    Returns:
        Redis client instance if Redis is configured, otherwise None.
    """
    redis_url = os.getenv("LLM_CACHE_REDIS_URL") or os.getenv("REDIS_URL")
    if not redis_url:
        return None

    global _CACHE_CLIENT, _CACHE_LOCK
    if _CACHE_CLIENT:
        return _CACHE_CLIENT

    if _CACHE_LOCK is None:
        _CACHE_LOCK = asyncio.Lock()

    lock = _CACHE_LOCK

    async with lock:
        if _CACHE_CLIENT:
            return _CACHE_CLIENT

        _CACHE_CLIENT = Redis.from_url(
            redis_url,
            encoding="utf-8",
            decode_responses=True,
        )
        return _CACHE_CLIENT


def _filter_cacheable_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Remove non-serializable items from kwargs for cache key construction.

    Args:
        kwargs: Parameters passed to the LLM call.

    Returns:
        Dictionary containing only serializable, non-sensitive values.
    """
    safe_keys: Dict[str, Any] = {}
    sensitive_keys = {"api_key", "authorization", "Authorization"}

    for key, value in kwargs.items():
        if key in sensitive_keys:
            continue

        if isinstance(value, (str, int, float, bool)) or value is None:
            safe_keys[key] = value
        elif isinstance(value, (list, dict)):
            safe_keys[key] = value
        else:
            safe_keys[key] = str(value)

    return safe_keys


def build_completion_cache_key(
    *,
    model: str,
    binding: str,
    base_url: Optional[str],
    system_prompt: str,
    prompt: str,
    messages: Optional[List[Dict[str, Any]]],
    kwargs: Dict[str, Any],
) -> str:
    """
    Construct a stable cache key for LLM completion calls.

    Args:
        model: Target model name.
        binding: Provider binding identifier.
        base_url: Provider base URL.
        system_prompt: System prompt used for the call.
        prompt: User prompt.
        messages: Optional OpenAI-style message array.
        kwargs: Remaining call parameters.

    Returns:
        Deterministic cache key string.
    """
    payload: Dict[str, Any] = {
        "model": model,
        "binding": binding,
        "base_url": base_url or "default",
        "system_prompt": system_prompt,
        "prompt": prompt,
        "messages": messages or [],
        "kwargs": _filter_cacheable_kwargs(kwargs),
    }

    try:
        serialized = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
    except TypeError:
        payload["kwargs"] = {k: str(v) for k, v in payload["kwargs"].items()}
        serialized = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )

    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return f"{CACHE_NAMESPACE}:{binding}:{model}:{digest}"


async def get_cached_completion(key: str) -> Optional[str]:
    """
    Fetch a cached completion if available.

    Args:
        key: Cache key to read.

    Returns:
        Cached completion text when present, otherwise None.
    """
    client = await get_cache_client()
    if not client:
        return None

    try:
        return await client.get(key)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Redis cache get failed: {exc}")
        return None


async def set_cached_completion(
    key: str, value: str, ttl_seconds: Optional[int] = None
) -> None:
    """
    Store a completion response in Redis.

    Args:
        key: Cache key to write.
        value: Completion text to store.
        ttl_seconds: Optional TTL override in seconds.
    """
    client = await get_cache_client()
    if not client:
        return

    ttl = ttl_seconds if ttl_seconds is not None else DEFAULT_CACHE_TTL
    try:
        await client.set(key, value, ex=ttl)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Redis cache set failed: {exc}")


__all__ = [
    "CACHE_NAMESPACE",
    "DEFAULT_CACHE_TTL",
    "build_completion_cache_key",
    "get_cached_completion",
    "get_cache_client",
    "set_cached_completion",
]
