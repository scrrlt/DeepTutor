# -*- coding: utf-8 -*-
"""
LLM Factory - Central Hub for LLM Calls
=======================================

This module serves as the central hub for all LLM calls in DeepTutor.
It provides a unified interface for agents to call LLMs, routing requests
to the appropriate provider (cloud or local) based on URL detection.

Architecture:
    Agents (ChatAgent, GuideAgent, etc.)
              ↓
         BaseAgent.call_llm() / stream_llm()
              ↓
         LLM Factory (this module)
              ↓
    ┌─────────┴─────────┐
    ↓                   ↓
CloudProvider      LocalProvider
(cloud_provider)   (local_provider)
              ↓                   ↓
OpenAI/DeepSeek/etc    LM Studio/Ollama/etc

Routing:
- Automatically routes to local_provider for local URLs (localhost, 127.0.0.1, etc.)
- Routes to cloud_provider for all other URLs

Retry Mechanism:
- Automatic retry with exponential backoff for transient errors
- Configurable max_retries, retry_delay, and exponential_backoff
- Only retries on retriable errors (timeout, rate limit, server errors)
"""

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

import tenacity

from src.logging.logger import Logger, get_logger

from . import local_provider
from .config import get_llm_config
from .exceptions import (
    LLMAPIError,
    LLMAuthenticationError,
    LLMRateLimitError,
    LLMTimeoutError,
)
from .utils import is_local_llm_server

# Initialize logger
logger: Logger = get_logger("LLMFactory")

# Default retry configuration
DEFAULT_MAX_RETRIES = 5  # Increased for complex agents like Research
DEFAULT_RETRY_DELAY = 2.0  # seconds
DEFAULT_EXPONENTIAL_BACKOFF = True

CallKwargs: TypeAlias = dict[str, Any]


def _is_retriable_error(error: Exception) -> bool:
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
    - Unknown API errors without status codes
    """
    from aiohttp import ClientError

    if isinstance(
        error,
        (
            asyncio.TimeoutError,
            ClientError,
            LLMTimeoutError,
            LLMRateLimitError,
        ),
    ):
        return True

    if isinstance(error, LLMAuthenticationError):
        return False  # Don't retry auth errors

    if isinstance(error, LLMAPIError):
        status_code = error.status_code
        if status_code is None:
            return False
        # Retry on server errors (5xx) and rate limits (429)
        if status_code >= 500 or status_code == 429:
            return True
        # Don't retry on client errors (4xx except 429)
        if 400 <= status_code < 500:
            return False
        return False

    return False


def _should_use_local(base_url: Optional[str]) -> bool:
    """
    Determine if we should use the local provider based on URL.

    Args:
        base_url: The base URL to check

    Returns:
        True if local provider should be used (localhost, 127.0.0.1, etc.)
    """
    return is_local_llm_server(base_url) if base_url else False


async def complete(
    prompt: str,
    system_prompt: str = "You are a helpful assistant.",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    api_version: Optional[str] = None,
    binding: Optional[str] = None,
    messages: Optional[List[Dict[str, str]]] = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_delay: float = DEFAULT_RETRY_DELAY,
    exponential_backoff: bool = DEFAULT_EXPONENTIAL_BACKOFF,
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
        max_retries: Maximum number of retry attempts (default: 5)
        retry_delay: Initial delay between retries in seconds (default: 2.0)
        exponential_backoff: Whether to use exponential backoff (default: True)
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

    def _log_retry_warning(retry_state: tenacity.RetryCallState) -> None:
        """
        Log retry warnings with safe handling for missing exceptions.

        Args:
            retry_state: Tenacity retry state for the current attempt.
        """
        outcome = retry_state.outcome
        if outcome is None:
            return

        error_message = "unknown error"
        try:
            exception = outcome.exception()
            if exception is not None:
                error_message = str(exception)
        except Exception:
            # Fail silently if we can't extract error message from exception
            # The error will still be raised, we just won't have a custom message
            pass  # nosec B110

        logger.warning(
            "LLM call failed (attempt %d/%d), retrying in %.1fs... Error: %s",
            retry_state.attempt_number,
            max_retries + 1,
            retry_state.upcoming_sleep or 0,
            error_message,
        )

    if exponential_backoff:
        # Use Any type annotation to avoid mypy issues with wait_exponential/wait_fixed union
        wait_strategy: Any = tenacity.wait_exponential(
            multiplier=retry_delay,
            min=retry_delay,
            max=120,
        )
    else:
        wait_strategy = tenacity.wait_fixed(retry_delay)

    def _log_retry_warning(retry_state: tenacity.RetryCallState) -> None:
        """
        Log retry warnings with safe handling for missing exceptions.

        Args:
            retry_state: Tenacity retry state for the current attempt.
        """
        outcome = retry_state.outcome
        if outcome is None:
            return

        error_message = "unknown error"
        try:
            exception = outcome.exception()
            if exception is not None:
                error_message = str(exception)
        except Exception:
            pass

        logger.warning(
            "LLM call failed (attempt %d/%d), retrying in %.1fs... Error: %s",
            retry_state.attempt_number,
            max_retries + 1,
            retry_state.upcoming_sleep or 0,
            error_message,
        )

    if exponential_backoff:
        wait_strategy = tenacity.wait_exponential(multiplier=retry_delay, min=retry_delay, max=60)
    else:
        wait_strategy = tenacity.wait_fixed(retry_delay)

    # Define the actual completion function with tenacity retry
    @tenacity.retry(
        # Use lambda wrapper to ensure exception type checking before calling _is_retriable_error
        # This prevents TypeError when non-Exception objects are passed to the retry logic
        retry=tenacity.retry_if_exception(
            lambda e: _is_retriable_error(e) if isinstance(e, Exception) else False
        ),
        wait=wait_strategy,
        stop=tenacity.stop_after_attempt(max_retries + 1),
        before_sleep=_log_retry_warning,
    )
    async def _do_complete(call_kwargs: CallKwargs) -> str:
        try:
            if use_local:
                return await local_provider.complete(**call_kwargs)
            else:

                return await cloud_provider.complete(**call_kwargs)
        except Exception as e:
            # Map raw SDK exceptions to unified exceptions for retry logic
            from .error_mapping import map_error

            provider_value = call_kwargs.get("binding")
            provider_name = provider_value if isinstance(provider_value, str) else "unknown"
            mapped_error = map_error(e, provider=provider_name)
            raise mapped_error from e

    # Build call kwargs
    call_kwargs: CallKwargs = {
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

    # Execute with retry (handled by tenacity decorator)
    return await _do_complete(call_kwargs)


async def stream(
    prompt: str,
    system_prompt: str = "You are a helpful assistant.",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    api_version: Optional[str] = None,
    binding: Optional[str] = None,
    messages: Optional[List[Dict[str, str]]] = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_delay: float = DEFAULT_RETRY_DELAY,
    exponential_backoff: bool = DEFAULT_EXPONENTIAL_BACKOFF,
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
        max_retries: Maximum number of retry attempts (default: 5)
        retry_delay: Initial delay between retries in seconds (default: 2.0)
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

    # Retry logic for streaming (retry on connection errors)
    # Total attempts = 1 initial + max_retries
    total_attempts = max_retries + 1
    last_exception = None
    delay = retry_delay
    max_delay = 120  # Cap maximum delay at 120 seconds (consistent with complete())

    chunks_yielded = False

    for attempt in range(total_attempts):
        try:
            # Route to appropriate provider
            if use_local:
                iterator = local_provider.stream(**call_kwargs)
            else:
                iterator = cloud_provider.stream(**call_kwargs)

            async for chunk in iterator:
                chunks_yielded = True
                yield chunk
            # If we get here, streaming completed successfully
            return
        except Exception as e:
            last_exception = e

            if chunks_yielded:
                logger.error(
                    "LLM streaming failed after yielding data; retry disabled.",
                    exc_info=True,
                )
                raise

            # Check if we should retry
            if attempt >= max_retries or not _is_retriable_error(e):
                raise

            # Calculate delay for next attempt
            if exponential_backoff:
                current_delay = min(delay * (2**attempt), max_delay)
            else:
                current_delay = delay

            # Special handling for rate limit errors with retry_after
            if isinstance(e, LLMRateLimitError) and e.retry_after:
                current_delay = max(current_delay, e.retry_after)

            # Log retry attempt (consistent with complete() function)
            logger.warning(
                "LLM streaming failed (attempt %d/%d), retrying in %.1fs... Error: %s",
                attempt + 1,
                total_attempts,
                current_delay,
                str(e),
            )

            # Wait before retrying
            await asyncio.sleep(current_delay)

    # Should not reach here, but just in case
    if last_exception:
        raise last_exception


async def fetch_models(
    binding: str,
    base_url: str,
    api_key: Optional[str] = None,
) -> List[str]:
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
    "openai": {
        "name": "OpenAI",
        "base_url": "https://api.openai.com/v1",
        "requires_key": True,
        "models": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
    },
    "anthropic": {  # Prefer ANTHROPIC_API_KEY; CLAUDE_API_KEY is legacy alias
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
    """
    Get all provider presets for frontend display.
    """
    return {
        "api": API_PROVIDER_PRESETS,
        "local": LOCAL_PROVIDER_PRESETS,
    }


__all__ = [
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
