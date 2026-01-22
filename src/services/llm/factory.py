"""LLM factory and routing helpers.

This module provides the public, backward-compatible LLM entry points
(`complete`, `stream`). Implementation is delegated to a provider object
so the factory stays focused on instantiation/config wiring.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import (
    Any,
    AsyncGenerator,
    Dict,
    List,
    Optional,
    TypeAlias,
    TypedDict,
)
from collections.abc import Mapping
from typing import (
    Any,
    AsyncGenerator,
    Dict,
    List,
    Optional,
    TypeAlias,
    TypedDict,
)

import tenacity

from src.config.settings import settings
from src.logging.logger import Logger, get_logger

from . import local_provider
from . import local_provider
from .config import get_llm_config
from .exceptions import (
    LLMAPIError,  # noqa: F401
    LLMAuthenticationError,  # noqa: F401
    LLMConfigError,  # noqa: F401
    LLMRateLimitError,  # noqa: F401
    LLMTimeoutError,  # noqa: F401
)
from .providers.base_provider import BaseLLMProvider
from .utils import is_local_llm_server

# Initialize logger
logger: Logger = get_logger("LLMFactory")
logger: Logger = get_logger("LLMFactory")

# Default retry configuration (bound to settings)
DEFAULT_MAX_RETRIES = settings.retry.max_retries
DEFAULT_RETRY_DELAY = settings.retry.base_delay
DEFAULT_EXPONENTIAL_BACKOFF = settings.retry.exponential_backoff

CallKwargs: TypeAlias = dict[str, Any]


def _is_retriable_error(error: BaseException) -> bool:
    """
    Check if an error is retriable.

    Retriable errors:
    - Timeout errors
    - Rate limit errors (429)
    - Server errors (5xx)
    - Network/connection errors

    Non-retriable errors:
    - Authentication errors (401)
    - Bad request (400)
    - Not found (404)
    - Client errors (4xx except 429)
    """
    from aiohttp import ClientError

    if isinstance(error, (asyncio.TimeoutError, ClientError)):
    if isinstance(error, (asyncio.TimeoutError, ClientError)):
        return True

    if isinstance(exc, LLMRateLimitError):
        return True

    if isinstance(exc, LLMAuthenticationError):
        return False

    if isinstance(error, LLMAPIError):
        status_code = error.status_code
        if status_code:
            # Retry on server errors (5xx) and rate limits (429)
            if status_code >= 500 or status_code == 429:
                return True
            # Don't retry on client errors (4xx except 429)
            if 400 <= status_code < 500:
                return False

        # FIX: If status_code is None (e.g. connection drop), RETRY.
        # This aligns with the catch-all "return True" at the end.
        return True

    # For other exceptions (network errors, etc.), retry
    return True


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
    **kwargs: object,
    **kwargs: object,
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
    if not model or not base_url:
        config = get_llm_config()
        model = model or config.model
        api_key = api_key if api_key is not None else config.api_key
        base_url = base_url or config.base_url
        api_version = api_version or config.api_version
        binding = binding or config.binding or "openai"

    # Determine which provider to use
    use_local = _should_use_local(base_url)

    # Define helper to determine if a generic LLMAPIError is retriable
    def _is_retriable_llm_api_error(exc: BaseException) -> bool:
        """
        Thin wrapper around the module-level _is_retriable_error helper.

        Keeps a local, semantically named helper within complete() while
        delegating the actual retriability logic to the shared function
        to avoid code duplication.
        """
        return _is_retriable_error(exc)

    def _log_retry_warning(retry_state: tenacity.RetryCallState) -> None:
        """
        Log retry warnings with safe handling for missing exceptions.

        Args:
            retry_state: Tenacity retry state for the current attempt.
        """
        outcome = retry_state.outcome
        exception = outcome.exception() if outcome else None
        error_message = str(exception) if exception else "unknown error"
        message = (
            "LLM call failed (attempt "
            f"{retry_state.attempt_number}/{max_retries + 1}), "
            f"retrying in {retry_state.upcoming_sleep:.1f}s... Error: "
            f"{error_message}"
        )
        logger.warning(message)

    if exponential_backoff:
        wait_strategy = tenacity.wait_exponential(multiplier=retry_delay, min=retry_delay, max=60)  # type: ignore[assignment]
    else:
        wait_strategy = tenacity.wait_fixed(retry_delay)  # type: ignore[assignment]

    # Define the actual completion function with tenacity retry
    @tenacity.retry(
        retry=(
            tenacity.retry_if_exception_type(LLMRateLimitError)
            | tenacity.retry_if_exception_type(LLMTimeoutError)
            | tenacity.retry_if_exception(_is_retriable_llm_api_error)
        ),
        wait=wait_strategy,
        stop=tenacity.stop_after_attempt(max_retries + 1),
        before_sleep=_log_retry_warning,
        wait=wait_strategy,
        stop=tenacity.stop_after_attempt(max_retries + 1),
        before_sleep=_log_retry_warning,
    )
    async def _do_complete(call_kwargs: CallKwargs) -> str:
    async def _do_complete(call_kwargs: CallKwargs) -> str:
        try:
            if use_local:
                return await local_provider.complete(**call_kwargs)
            else:
                from . import cloud_provider

                from . import cloud_provider

                return await cloud_provider.complete(**call_kwargs)
        except Exception as e:
            # Map raw SDK exceptions to unified exceptions for retry logic
            from .error_mapping import map_error

            provider_value = call_kwargs.get("binding")
            provider_name = provider_value if isinstance(provider_value, str) else "unknown"
            mapped_error = map_error(e, provider=provider_name)
            provider_value = call_kwargs.get("binding")
            provider_name = provider_value if isinstance(provider_value, str) else "unknown"
            mapped_error = map_error(e, provider=provider_name)
            raise mapped_error from e

    # Build call kwargs
    call_kwargs: CallKwargs = {
    call_kwargs: CallKwargs = {
        "prompt": prompt,
        "system_prompt": system_prompt,
        "model": model,
        "api_key": api_key,
        "base_url": base_url,
        **kwargs,
    }

    # Only include messages if it's not None
    if messages is not None:
        call_kwargs["messages"] = messages

    # Add cloud-specific kwargs if not local
    if not use_local:
        call_kwargs["api_version"] = api_version
        call_kwargs["binding"] = binding or "openai"

    # Execute with retry (handled by tenacity decorator)
    return await _do_complete(call_kwargs)
    return await _do_complete(call_kwargs)


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
    **kwargs: object,
    **kwargs: object,
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
        max_retries: Maximum number of retry attempts (default: 3)
        retry_delay: Initial delay between retries in seconds (default: 1.0)
        exponential_backoff: Whether to use exponential backoff (default: True)
        **kwargs: Additional parameters (temperature, max_tokens, etc.)

    Yields:
        str: Response chunks
    """
    # Get config if parameters not provided
    if not model or not base_url:
        config = get_llm_config()
        model = model or config.model
        api_key = api_key if api_key is not None else config.api_key
        base_url = base_url or config.base_url
        api_version = api_version or config.api_version
        binding = binding or config.binding or "openai"

    # Determine which provider to use
    use_local = _should_use_local(base_url)

    # Build call kwargs
    call_kwargs: CallKwargs = {
    call_kwargs: CallKwargs = {
        "prompt": prompt,
        "system_prompt": system_prompt,
        "model": model,
        "api_key": api_key,
        "base_url": base_url,
        **kwargs,
    }

    # Only include messages if it's not None
    if messages is not None:
        call_kwargs["messages"] = messages

    # Add cloud-specific kwargs if not local
    if not use_local:
        call_kwargs["api_version"] = api_version
        call_kwargs["binding"] = binding or "openai"

    # Retry logic for streaming (retry on connection errors)
    last_exception = None
    delay = retry_delay
    has_yielded = False
    has_yielded = False

    for attempt in range(max_retries + 1):
    for attempt in range(max_retries + 1):
        try:
            # Route to appropriate provider
            if use_local:
                async for chunk in local_provider.stream(**call_kwargs):
                    has_yielded = True
                    has_yielded = True
                    yield chunk
            else:
                from . import cloud_provider

                from . import cloud_provider

                async for chunk in cloud_provider.stream(**call_kwargs):
                    has_yielded = True
                    has_yielded = True
                    yield chunk
            # If we get here, streaming completed successfully
            return
        except Exception as e:
            last_exception = e

            # If we've already yielded, don't retry
            if has_yielded:
                raise

            # If we've already yielded, don't retry
            if has_yielded:
                raise

            # Check if we should retry
            if attempt >= max_retries or not _is_retriable_error(e):
                raise

            # Calculate delay for next attempt
            if exponential_backoff:
                current_delay = min(delay * (2**attempt), 60)  # Cap at 60 seconds
                current_delay = min(delay * (2**attempt), 60)  # Cap at 60 seconds
            else:
                current_delay = delay

            # Special handling for rate limit errors with retry_after
            if isinstance(e, LLMRateLimitError) and e.retry_after:
                current_delay = max(current_delay, e.retry_after)

            # Wait before retrying
            await asyncio.sleep(current_delay)

    # Should not reach here, but just in case
    if last_exception:
        raise last_exception


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
        from . import cloud_provider

        from . import cloud_provider

        return await cloud_provider.fetch_models(base_url, api_key, binding)


class ApiProviderPreset(TypedDict, total=False):
    """Typed representation of API provider presets."""

    name: str
    base_url: str
    requires_key: bool
    models: list[str]
    binding: str


class LocalProviderPreset(TypedDict, total=False):
    """Typed representation of local provider presets."""

    name: str
    base_url: str
    requires_key: bool
    default_key: str


ProviderPreset: TypeAlias = ApiProviderPreset | LocalProviderPreset
ProviderPresetMap: TypeAlias = Mapping[str, ProviderPreset]
ProviderPresetBundle: TypeAlias = Mapping[str, ProviderPresetMap]


class ApiProviderPreset(TypedDict, total=False):
    """Typed representation of API provider presets."""

    name: str
    base_url: str
    requires_key: bool
    models: list[str]
    binding: str


class LocalProviderPreset(TypedDict, total=False):
    """Typed representation of local provider presets."""

    name: str
    base_url: str
    requires_key: bool
    default_key: str


ProviderPreset: TypeAlias = ApiProviderPreset | LocalProviderPreset
ProviderPresetMap: TypeAlias = Mapping[str, ProviderPreset]
ProviderPresetBundle: TypeAlias = Mapping[str, ProviderPresetMap]


# API Provider Presets
API_PROVIDER_PRESETS: dict[str, ApiProviderPreset] = {
API_PROVIDER_PRESETS: dict[str, ApiProviderPreset] = {
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
LOCAL_PROVIDER_PRESETS: dict[str, LocalProviderPreset] = {
LOCAL_PROVIDER_PRESETS: dict[str, LocalProviderPreset] = {
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


def get_provider_presets() -> ProviderPresetBundle:
def get_provider_presets() -> ProviderPresetBundle:
    """
    Get all provider presets for frontend display.
    """
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
