"""LLM factory and routing helpers.

This module provides the public, backward-compatible LLM entry points
(`complete`, `stream`). Implementation is delegated to a provider object
so the factory stays focused on instantiation/config wiring.
"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncGenerator, Awaitable, Callable

from src.logging.logger import get_logger

from .config import LLMConfig, get_llm_config
from .exceptions import (
    LLMAPIError,  # noqa: F401
    LLMAuthenticationError,  # noqa: F401
    LLMConfigError,  # noqa: F401
    LLMRateLimitError,  # noqa: F401
    LLMTimeoutError,  # noqa: F401
)
from .providers.base_provider import BaseLLMProvider
from . import cloud_provider, local_provider, registry
from .utils import is_local_llm_server

# Initialize logger
logger = get_logger("LLMFactory")

# Default retry configuration (kept for public API compatibility)
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 1.0  # seconds
DEFAULT_EXPONENTIAL_BACKOFF = True


class LLMFactory:
    """Instantiate providers from validated configuration."""

    @staticmethod
    def get_provider(config: LLMConfig) -> BaseLLMProvider:
        """Create a provider instance for the given config."""
        # Ensure the routing provider is registered without importing heavy SDKs.
        from .providers.routing import RoutingProvider  # noqa: F401

        provider_cls = registry.get_provider_class("routing")
        provider = provider_cls(config)
        if isinstance(provider, BaseLLMProvider):
            return provider

        class _ProviderAdapter(BaseLLMProvider):
            def __init__(self, cfg: LLMConfig, wrapped: Any) -> None:
                self.config = cfg
                self.provider_name = getattr(cfg, "provider_name", getattr(cfg, "binding", ""))
                try:
                    self.api_key = getattr(cfg, "get_api_key", lambda: None)()
                except Exception:
                    self.api_key = None
                self.base_url = getattr(cfg, "base_url", None)
                self._wrapped = wrapped

            async def complete(self, prompt: str, **kwargs: Any):
                return await self._wrapped.complete(prompt, **kwargs)

            async def stream(self, prompt: str, **kwargs: Any):
                return await self._wrapped.stream(prompt, **kwargs)

        return _ProviderAdapter(config, provider)


def _apply_config_overrides(
    config: LLMConfig, updates: dict[str, Any]
) -> LLMConfig:
    """Apply overrides with validation.

    Pydantic's `model_copy(update=...)` does not validate/coerce updated
    fields. For settings models (notably `SecretStr`), we need to re-validate.
    """
    if not updates:
        return config
    data = config.model_dump()
    data.update(updates)
    return LLMConfig.model_validate(data)


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
    # Resolve validated config.
    if config is None:
        if model is not None:
            effective_config = LLMConfig(
                model=model,
                binding=binding or "openai",
                base_url=base_url,
                api_key=api_key,
                api_version=api_version,
            )
        else:
            try:
                effective_config = get_llm_config()
            except Exception:
                effective_config = LLMConfig(
                    model="gpt-4o-mini",
                    binding=binding or "openai",
                    base_url=base_url,
                    api_key=api_key,
                    api_version=api_version,
                )
    else:
        effective_config = config

    # Apply explicit overrides (even when starting from get_llm_config()).
    updates: dict[str, Any] = {}
    if model is not None:
        updates["model"] = model
    if base_url is not None:
        updates["base_url"] = base_url
    if binding is not None:
        updates["binding"] = binding
    if api_key is not None:
        updates["api_key"] = api_key
    if api_version is not None:
        updates["api_version"] = api_version

    effective_config = _apply_config_overrides(effective_config, updates)

    provider = LLMFactory.get_provider(effective_config)

    response = await provider.complete(
        prompt,
        system_prompt=system_prompt,
        messages=messages,
        max_retries=max_retries,
        sleep=sleep,
        use_cache=use_cache,
        cache_ttl_seconds=cache_ttl_seconds,
        cache_key=cache_key,
        # Kept for API compatibility; provider-level backoff controls.
        retry_delay=retry_delay,
        exponential_backoff=exponential_backoff,
        **kwargs,
    )

    return response.content


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
    if config is None:
        if model is not None:
            effective_config = LLMConfig(
                model=model,
                binding=binding or "openai",
                base_url=base_url,
                api_key=api_key,
                api_version=api_version,
            )
        else:
            try:
                effective_config = get_llm_config()
            except Exception:
                effective_config = LLMConfig(
                    model="gpt-4o-mini",
                    binding=binding or "openai",
                    base_url=base_url,
                    api_key=api_key,
                    api_version=api_version,
                )
    else:
        effective_config = config

    updates: dict[str, Any] = {}
    if model is not None:
        updates["model"] = model
    if base_url is not None:
        updates["base_url"] = base_url
    if binding is not None:
        updates["binding"] = binding
    if api_key is not None:
        updates["api_key"] = api_key
    if api_version is not None:
        updates["api_version"] = api_version

    effective_config = _apply_config_overrides(effective_config, updates)

    provider = LLMFactory.get_provider(effective_config)

    async for chunk in provider.stream(
        prompt,
        system_prompt=system_prompt,
        messages=messages,
        max_retries=max_retries,
        # Kept for API compatibility; provider-level backoff controls.
        retry_delay=retry_delay,
        exponential_backoff=exponential_backoff,
        **kwargs,
    ):
        if chunk.delta:
            yield chunk.delta


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
