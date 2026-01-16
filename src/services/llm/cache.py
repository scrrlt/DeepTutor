"""Redis-backed cache helpers for LLM completions."""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
from typing import Any

from redis.asyncio import Redis

from src.logging import get_logger

logger = get_logger("LLMCache")

# Global state
_CACHE_CLIENT: Redis | None = None
_CACHE_LOCK: asyncio.Lock | None = None
CACHE_NAMESPACE = os.getenv("LLM_CACHE_NAMESPACE", "llm_cache")


def _get_default_cache_ttl() -> int:
    """
    Safely parse cache TTL from environment.

    Returns:
        The cache TTL in seconds (default: 3600).
    """
    raw_ttl = os.getenv("LLM_CACHE_TTL", "3600")
    try:
        return int(raw_ttl)
    except ValueError:
        logger.warning(
            f"Invalid LLM_CACHE_TTL '{raw_ttl}', defaulting to 3600s"
        )
        return 3600


DEFAULT_CACHE_TTL = _get_default_cache_ttl()


async def get_cache_client() -> Redis | None:
    """
    Get or create a Redis client.

    Thread-safe and Event-Loop-safe singleton pattern.

    Returns:
        The initialized Redis client or None if connection fails.
    """
    redis_url = os.getenv("LLM_CACHE_REDIS_URL") or os.getenv("REDIS_URL")
    if not redis_url:
        return None

    global _CACHE_CLIENT, _CACHE_LOCK

    # Fast path: Client exists
    if _CACHE_CLIENT is not None:
        return _CACHE_CLIENT

    # Lazy init lock to ensure it binds to the CURRENT event loop
    if _CACHE_LOCK is None:
        _CACHE_LOCK = asyncio.Lock()

    async with _CACHE_LOCK:
        # Double-check inside lock
        if _CACHE_CLIENT is not None:
            return _CACHE_CLIENT

        try:
            # Log host only, hide auth
            sanitized_url = redis_url.split("@")[-1]
            logger.debug(f"Connecting to Redis at {sanitized_url}")
            _CACHE_CLIENT = Redis.from_url(
                redis_url,
                encoding="utf-8",
                decode_responses=True,
                socket_connect_timeout=2.0,  # Fail fast
                socket_keepalive=True,
            )
            # Test connection
            await _CACHE_CLIENT.ping()
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            _CACHE_CLIENT = None
            return None

        return _CACHE_CLIENT


def _sanitize_kwargs_for_hashing(kwargs: dict[str, Any]) -> dict[str, Any]:
    """
    Prepare kwargs for hashing.

    1. Filter known sensitive keys.
    2. Sort keys for deterministic hashing.
    3. Convert non-primitives to strings to ensure serializability.

    Args:
        kwargs: The dictionary of arguments to sanitize.

    Returns:
        A new dictionary with sanitized values.
    """
    # Expanded blocklist
    sensitive_patterns = {
        "api_key",
        "key",
        "token",
        "secret",
        "password",
        "auth",
        "authorization",
        "credential",
    }

    sanitized = {}
    for k, v in kwargs.items():
        # Case-insensitive check against sensitive patterns
        if k.lower() in sensitive_patterns:
            continue

        # Recursive cleaning could go here, but flat processing is usually sufficient for kwargs
        if isinstance(v, (str, int, float, bool, type(None))):
            sanitized[k] = v
        else:
            # Force string representation for complex objects (Lists, Dicts, Custom Objects)
            # This avoids the "try/except JSON" loop later
            sanitized[k] = str(v)

    return sanitized


def build_completion_cache_key(
    *,
    model: str,
    binding: str,
    base_url: str | None,
    system_prompt: str,
    prompt: str,
    messages: list[dict[str, Any]] | None,
    kwargs: dict[str, Any],
) -> str:
    """
    Construct a deterministic SHA-256 cache key.

    Args:
        model: The model identifier.
        binding: The provider binding name.
        base_url: The API base URL.
        system_prompt: The system prompt.
        prompt: The user prompt.
        messages: List of message dictionaries.
        kwargs: Additional generation arguments.

    Returns:
        A namespaced cache key string.
    """
    # 1. Sanitize kwargs immediately
    clean_kwargs = _sanitize_kwargs_for_hashing(kwargs)

    # 2. Build deterministic payload
    payload = {
        "model": model,
        "binding": binding,
        "base_url": base_url or "default",
        "system_prompt": system_prompt,
        "prompt": prompt,
        "messages": messages or [],
        "kwargs": clean_kwargs,
    }

    # 3. Serialize with sort_keys=True for stability
    # We don't need a try/except block here because _sanitize_kwargs_for_hashing
    # already coerced complex types to strings.
    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),  # Compact representation
        ensure_ascii=False,  # Allow Unicode in cache keys
    )

    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return f"{CACHE_NAMESPACE}:{binding}:{model}:{digest}"


async def get_cached_completion(key: str) -> str | None:
    """
    Fetch from Redis, handling connection errors gracefully.

    Args:
        key: The cache key to retrieve.

    Returns:
        The cached completion string or None if not found/error.
    """
    client = await get_cache_client()
    if not client:
        return None

    try:
        # We don't lock for reads; Redis is thread-safe
        return await client.get(key)
    except Exception as exc:
        # Don't crash the LLM pipeline just because cache is flaky
        logger.warning(f"Cache read error: {exc}")
        return None


async def set_cached_completion(
    key: str, value: str, ttl_seconds: int | None = None
) -> None:
    """
    Write to Redis, handling connection errors gracefully.

    Args:
        key: The cache key to write.
        value: The completion string to cache.
        ttl_seconds: Optional TTL in seconds (overrides default).
    """
    client = await get_cache_client()
    if not client:
        return

    ttl = ttl_seconds if ttl_seconds is not None else DEFAULT_CACHE_TTL
    try:
        # Fire and forget (await, but don't block logic if it fails)
        await client.set(key, value, ex=ttl)
    except Exception as exc:
        logger.warning(f"Cache write error: {exc}")


__all__ = [
    "CACHE_NAMESPACE",
    "DEFAULT_CACHE_TTL",
    "build_completion_cache_key",
    "get_cached_completion",
    "get_cache_client",
    "set_cached_completion",
]
