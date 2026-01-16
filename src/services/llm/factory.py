"""LLM factory and routing helpers.

Provides unified entry points for cloud and local LLM calls with retry logic.
"""

import asyncio
import json
import os
from functools import partial
from typing import Any, AsyncGenerator, Awaitable, Callable

import tenacity

from src.logging.logger import get_logger

from . import cloud_provider, local_provider
from .cache import (
    DEFAULT_CACHE_TTL,
    build_completion_cache_key,
    get_cached_completion,
    set_cached_completion,
)
from .config import LLMConfig, get_llm_config
from .exceptions import (
    LLMAPIError,
    LLMAuthenticationError,
    LLMConfigError,
    LLMRateLimitError,
    LLMTimeoutError,
)
from .utils import is_local_llm_server

# Initialize logger
logger = get_logger("LLMFactory")

# Default retry configuration
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 1.0  # seconds
DEFAULT_EXPONENTIAL_BACKOFF = True


def _should_retry_error(exc: BaseException) -> bool:
    """Single source of truth for retry logic."""
    if isinstance(exc, (ConnectionError, asyncio.TimeoutError, LLMTimeoutError)):
        return True

    if isinstance(exc, LLMRateLimitError):
        return True

    if isinstance(exc, LLMAuthenticationError):
        return False

    if isinstance(exc, LLMAPIError):
        status_code = exc.status_code
        if status_code is None:
            return False
        if status_code == 429 or status_code >= 500:
            return True
        if 400 <= status_code < 500:
            return False

    return False


def _is_retriable_error(exc: BaseException) -> bool:
    """Backward-compatible retry predicate used by legacy tests."""
    return _should_retry_error(exc)


def _is_retriable_llm_api_error(error: LLMAPIError) -> bool:
    """Compatibility helper for tests expecting API retry predicate."""
    return _should_retry_error(error)


def _should_use_local(base_url: str | None) -> bool:
    """
    Determine if we should use the local provider based on URL.

    Args:
        base_url: The base URL to check

    Returns:
        True if local provider should be used (localhost, 127.0.0.1, etc.)
    """
    return is_local_llm_server(base_url) if base_url else False


def _sanitize_cache_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Strip non-cacheable keys before hashing using an allowlist."""
    allowlist = {
        "prompt",
        "system_prompt",
        "model",
        "binding",
        "base_url",
        "messages",
        "temperature",
        "max_tokens",
        "max_completion_tokens",
        "top_p",
        "presence_penalty",
        "frequency_penalty",
        "stop",
        "timeout",
    }
    cacheable: dict[str, Any] = {}
    for key in allowlist:
        if key in kwargs:
            cacheable[key] = kwargs[key]
    return cacheable


def _build_retrying(
    *,
    max_retries: int,
    retry_delay: float,
    exponential_backoff: bool,
    sleep: Callable[[float], Awaitable[None] | None] | None,
) -> tenacity.AsyncRetrying:
    wait_strategy: Any
    if exponential_backoff:
        wait_strategy = tenacity.wait_exponential(
            multiplier=retry_delay,
            min=retry_delay,
            max=60,
        )
    else:
        wait_strategy = tenacity.wait_fixed(retry_delay)

    def _log_retry(retry_state: tenacity.RetryCallState) -> None:
        outcome_exception = None
        if retry_state.outcome is not None:
            outcome_exception = retry_state.outcome.exception()

        message = (
            "LLM call failed (attempt %d/%d), retrying in %.1fs... Error: %s"
            % (
                retry_state.attempt_number,
                max_retries + 1,
                retry_state.upcoming_sleep,
                outcome_exception,
            )
        )
        logger.warning(message)

    retry_kwargs: dict[str, Any] = {
        "retry": tenacity.retry_if_exception(_should_retry_error),
        "wait": wait_strategy,
        "stop": tenacity.stop_after_attempt(max_retries + 1),
        "before_sleep": _log_retry,
    }
    if sleep is not None:
        retry_kwargs["sleep"] = sleep

    return tenacity.AsyncRetrying(**retry_kwargs)


async def _invoke_provider(
    use_local: bool, call_kwargs: dict[str, Any]
) -> str:
    if use_local:
        return await local_provider.complete(**call_kwargs)
    return await cloud_provider.complete(**call_kwargs)


async def _execute_with_retry(
    call: Callable[[], Awaitable[str]], retrying: tenacity.AsyncRetrying
) -> str:
    """Run a call with tenacity retrying wrapper."""
    async for attempt in retrying:
        with attempt:
            return await call()

    raise RuntimeError("Retrying produced no attempts")


async def complete(
    prompt: str,
    system_prompt: str = "You are a helpful assistant.",
    config: LLMConfig | None = None,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    api_version: str | None = None,
    binding: str | None = None,
    messages: list[dict[str, str]] | None = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_delay: float = DEFAULT_RETRY_DELAY,
    exponential_backoff: bool = DEFAULT_EXPONENTIAL_BACKOFF,
    sleep: Callable[[float], Awaitable[None] | None] | None = None,
    use_cache: bool = True,
    cache_ttl_seconds: int | None = None,
    cache_key: str | None = None,
    **kwargs: Any,
) -> str:
    """
    Unified LLM completion function with automatic retry.

    Routes to cloud_provider or local_provider based on configuration.
    Includes automatic retry with exponential backoff for transient errors.

    Args:
        prompt: The user prompt
        system_prompt: System prompt for context
        model: Model name (optional, uses effective config if not provided)
        api_key: API key (optional)
        base_url: Base URL for the API (optional)
        api_version: API version for Azure OpenAI (optional)
        binding: Provider binding type (optional)
        messages: Pre-built messages array (optional)
        max_retries: Maximum number of retry attempts (default: 3)
        retry_delay: Initial delay between retries in seconds (default: 1.0)
        exponential_backoff: Whether to use exponential backoff (default: True)
        sleep: Optional sleep hook for testing.
        use_cache: Whether to use Redis-backed cache for completions.
        cache_ttl_seconds: Optional TTL override for cached responses.
        cache_key: Optional precomputed cache key.
        **kwargs: Additional parameters (temperature, max_tokens, etc.)

    Returns:
        str: The LLM response
    """
    # Get config if parameters not provided
    effective_config = config or None
    if effective_config is None and (not model or not base_url):
        effective_config = get_llm_config()

    if effective_config is not None:
        model = model or effective_config.model
        api_key = api_key if api_key is not None else effective_config.api_key
        base_url = base_url or effective_config.base_url
        api_version = api_version or effective_config.api_version
        binding = binding or effective_config.binding or "openai"

    # Determine which provider to use
    use_local = _should_use_local(base_url)

    if not use_local and not api_key:
        raise LLMConfigError("API key is required for cloud providers")

    # Build call kwargs
    call_kwargs = {
        "prompt": prompt,
        "system_prompt": system_prompt,
        "model": model,
        "api_key": api_key,
        "base_url": base_url,
        "messages": messages,
        **kwargs,
    }

    # Add cloud-specific kwargs if not local
    if not use_local:
        call_kwargs["api_version"] = api_version
        call_kwargs["binding"] = binding or "openai"

    cache_key_value: str | None = None
    if use_cache:
        cache_key_value = cache_key or build_completion_cache_key(
            model=model,
            binding=(binding or "openai"),
            base_url=base_url,
            system_prompt=system_prompt,
            prompt=prompt,
            messages=messages,
            kwargs=_sanitize_cache_kwargs(call_kwargs),
        )
        cached = await get_cached_completion(cache_key_value)
        if cached is not None:
            return cached

    retrying = _build_retrying(
        max_retries=max_retries,
        retry_delay=retry_delay,
        exponential_backoff=exponential_backoff,
        sleep=sleep,
    )

    result = await _execute_with_retry(
        partial(_invoke_provider, use_local, call_kwargs),
        retrying,
    )

    if use_cache and cache_key_value:
        await set_cached_completion(
            cache_key_value,
            result,
            ttl_seconds=cache_ttl_seconds or DEFAULT_CACHE_TTL,
        )

    return result


async def stream(
    prompt: str,
    system_prompt: str = "You are a helpful assistant.",
    config: LLMConfig | None = None,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    api_version: str | None = None,
    binding: str | None = None,
    messages: list[dict[str, str]] | None = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_delay: float = DEFAULT_RETRY_DELAY,
    exponential_backoff: bool = DEFAULT_EXPONENTIAL_BACKOFF,
    **kwargs: Any,
) -> AsyncGenerator[str, None]:
    """
    Unified LLM streaming function with automatic retry.

    Routes to cloud_provider or local_provider based on configuration.
    Includes automatic retry with exponential backoff for connection errors.

    Note: Retry only applies to initial connection errors. Once streaming
    starts, errors during streaming will not be automatically retried.

    Args:
        prompt: The user prompt
        system_prompt: System prompt for context
        model: Model name (optional, uses effective config if not provided)
        api_key: API key (optional)
        base_url: Base URL for the API (optional)
        api_version: API version for Azure OpenAI (optional)
        binding: Provider binding type (optional)
        messages: Pre-built messages array (optional)
        max_retries: Maximum number of retry attempts (default: 3)
        retry_delay: Initial delay between retries in seconds (default: 1.0)
        exponential_backoff: Whether to use exponential backoff (default: True)
        **kwargs: Additional parameters (temperature, max_tokens, etc.)

    Yields:
        str: Response chunks
    """
    # Get config if parameters not provided
    effective_config = config or None
    if effective_config is None and (not model or not base_url):
        effective_config = get_llm_config()

    if effective_config is not None:
        model = model or effective_config.model
        api_key = api_key if api_key is not None else effective_config.api_key
        base_url = base_url or effective_config.base_url
        api_version = api_version or effective_config.api_version
        binding = binding or effective_config.binding or "openai"

    # Determine which provider to use
    use_local = _should_use_local(base_url)

    # Build call kwargs
    call_kwargs = {
        "prompt": prompt,
        "system_prompt": system_prompt,
        "model": model,
        "api_key": api_key,
        "base_url": base_url,
        "messages": messages,
        **kwargs,
    }

    # Add cloud-specific kwargs if not local
    if not use_local:
        call_kwargs["api_version"] = api_version
        call_kwargs["binding"] = binding or "openai"

    retrying = _build_retrying(
        max_retries=max_retries,
        retry_delay=retry_delay,
        exponential_backoff=exponential_backoff,
        sleep=None,
    )

    def _get_stream_iterator() -> AsyncGenerator[str, None]:
        if use_local:
            return local_provider.stream(**call_kwargs)
        return cloud_provider.stream(**call_kwargs)

    iterator: AsyncGenerator[str, None] | None = None
    first_chunk: str | None = None
    async for attempt in retrying:
        with attempt:
            iterator = _get_stream_iterator()
            try:
                first_chunk = await iterator.__anext__()
            except StopAsyncIteration:
                return
            break

    if iterator is None:
        raise RuntimeError("Failed to establish stream iterator")

    if first_chunk is not None:
        yield first_chunk

    try:
        async for chunk in iterator:
            yield chunk
    except Exception as exc:
        logger.error(
            f"Stream interrupted during consumption: {exc}"
        )
        raise


async def fetch_models(
    binding: str,
    base_url: str,
    api_key: str | None = None,
) -> list[str]:
    """
    Fetch available models from the provider.

    Routes to cloud_provider or local_provider based on URL.

    Args:
        binding: Provider type (openai, ollama, etc.)
        base_url: API endpoint URL
        api_key: API key (optional for local providers)

    Returns:
        List of available model names
    """
    if is_local_llm_server(base_url):
        return await local_provider.fetch_models(base_url, api_key)
    else:
        return await cloud_provider.fetch_models(base_url, api_key, binding)


# API Provider Presets
API_PROVIDER_PRESETS = {
    "openai": {
        "name": "OpenAI",
        "base_url": "https://api.openai.com/v1",
        "requires_key": True,
        "models": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
    },
    "anthropic": {
        "name": "Anthropic",
        "base_url": "https://api.anthropic.com/v1",
        "requires_key": True,
        "binding": "anthropic",
        "models": ["claude-3-5-sonnet-20241022", "claude-3-haiku-20240307"],
    },
    "deepseek": {
        "name": "DeepSeek",
        "base_url": "https://api.deepseek.com",
        "requires_key": True,
        "models": ["deepseek-chat", "deepseek-reasoner"],
    },
    "openrouter": {
        "name": "OpenRouter",
        "base_url": "https://openrouter.ai/api/v1",
        "requires_key": True,
        "models": [],  # Dynamic
    },
}

# Local Provider Presets
LOCAL_PROVIDER_PRESETS = {
    "ollama": {
        "name": "Ollama",
        "base_url": "http://localhost:11434/v1",
        "requires_key": False,
        "default_key": "ollama",
    },
    "lm_studio": {
        "name": "LM Studio",
        "base_url": "http://localhost:1234/v1",
        "requires_key": False,
        "default_key": "lm-studio",
    },
    "vllm": {
        "name": "vLLM",
        "base_url": "http://localhost:8000/v1",
        "requires_key": False,
        "default_key": "vllm",
    },
    "llama_cpp": {
        "name": "llama.cpp",
        "base_url": "http://localhost:8080/v1",
        "requires_key": False,
        "default_key": "llama-cpp",
    },
}


def get_provider_presets() -> dict[str, Any]:
    """Get all provider presets for frontend display."""
    return {
        "api": API_PROVIDER_PRESETS,
        "local": LOCAL_PROVIDER_PRESETS,
    }


__all__ = [
    "_is_retriable_error",
    "_is_retriable_llm_api_error",
    "complete",
    "stream",
    "fetch_models",
    "get_provider_presets",
    "API_PROVIDER_PRESETS",
    "LOCAL_PROVIDER_PRESETS",
    # Retry configuration defaults
    "DEFAULT_MAX_RETRIES",
    "DEFAULT_RETRY_DELAY",
    "DEFAULT_EXPONENTIAL_BACKOFF",
]
