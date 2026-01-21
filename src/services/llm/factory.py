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

from src.config.settings import settings
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

# Default retry configuration (bound to settings)
DEFAULT_MAX_RETRIES = settings.retry.max_retries
DEFAULT_RETRY_DELAY = settings.retry.base_delay
DEFAULT_EXPONENTIAL_BACKOFF = settings.retry.exponential_backoff

CallKwargs: TypeAlias = dict[str, Any]


def _is_retriable_error(error: BaseException) -> bool:
    """
    Determine whether an exception should be treated as retriable for LLM provider calls.
    
    Recognizes timeouts, network/connection errors, rate limits, server errors, and unknown API errors (e.g., LLMAPIError with no status) as retriable; treats authentication errors and client 4xx errors (except 429) as non-retriable.
    
    Returns:
        `true` if the error is retriable, `false` otherwise.
    """
    from aiohttp import ClientError

    if isinstance(error, (asyncio.TimeoutError, ClientError)):
        return True
    if isinstance(error, LLMTimeoutError):
        return True
    if isinstance(error, LLMRateLimitError):
        return True
    if isinstance(error, LLMAuthenticationError):
        return False  # Don't retry auth errors

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
    Request a text completion from the configured LLM provider, automatically routing to a local or cloud provider and retrying transient errors.
    
    Parameters:
    	prompt (str): The user prompt to complete.
    	system_prompt (str): System prompt providing assistant context.
    	model (Optional[str]): Model name; uses configured default if omitted.
    	api_key (Optional[str]): API key to use for cloud providers.
    	base_url (Optional[str]): Base URL for the provider; determines local vs cloud routing.
    	api_version (Optional[str]): Cloud provider API version (used when not using a local provider).
    	binding (Optional[str]): Provider binding identifier (used when not using a local provider).
    	messages (Optional[List[Dict[str, str]]]): Pre-built message list to send instead of a single prompt.
    	max_retries (int): Maximum number of retry attempts for transient errors.
    	retry_delay (float): Initial retry delay in seconds; used as multiplier for exponential backoff when enabled.
    	exponential_backoff (bool): Whether to use exponential backoff between retries.
    	**kwargs (object): Additional provider-specific call options (e.g., temperature, max_tokens).
    
    Returns:
    	The final completion text returned by the LLM.
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
        Determine whether the given exception should trigger a retry for LLM API calls.
        
        Returns:
            bool: True if the exception is considered retriable, False otherwise.
        """
        return _is_retriable_error(exc)
    def _log_retry_warning(retry_state: tenacity.RetryCallState) -> None:
        """
        Log a formatted warning about a retry attempt including attempt count, upcoming sleep, and the failure reason.
        
        Parameters:
        	retry_state (tenacity.RetryCallState): Current tenacity retry state containing the attempt number, upcoming sleep duration, and outcome (exception or result). The logged message will include the exception message if present or "unknown error" otherwise.
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
    )
    async def _do_complete(call_kwargs: CallKwargs) -> str:
        """
        Invoke the selected provider's completion endpoint with the provided call arguments and return the resulting text.
        
        Parameters:
            call_kwargs (dict): Provider-specific keyword arguments forwarded to the provider's `complete` function.
        
        Returns:
            str: The completion text produced by the provider.
        
        Raises:
            Exception: A mapped, unified provider error if the underlying provider SDK raises an exception.
        """
        try:
            if use_local:
                return await local_provider.complete(**call_kwargs)
            else:
                from . import cloud_provider

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
    Stream model-generated text chunks for a prompt, routing to a local or cloud provider and retrying initial connections on transient errors.
    
    Streams response chunks produced for the given prompt. If model/base_url are omitted, the effective configuration is used. The function chooses a local provider when the base URL indicates a local LLM server; otherwise it calls the cloud provider and adds cloud-specific fields. Retries apply only to failures that occur before any chunk has been yielded; once streaming has started, errors will not be retried. Retry behavior is controlled by max_retries, retry_delay, and exponential_backoff; rate-limit errors with a retry_after value will use that delay when larger than the computed backoff.
    
    Parameters:
        prompt: The user prompt to generate a response for.
        system_prompt: System prompt providing assistant context.
        model: Optional model identifier; falls back to configured model when omitted.
        api_key: Optional API key for the provider.
        base_url: Optional base URL for the provider; determines local vs cloud routing when provided.
        api_version: Optional cloud API version (added when using a cloud provider).
        binding: Optional provider binding (added when using a cloud provider).
        messages: Optional list of pre-built message dicts to send instead of composing from prompt/system_prompt.
        max_retries: Maximum number of retry attempts for initial connection failures.
        retry_delay: Initial delay (in seconds) between retry attempts.
        exponential_backoff: If true, use exponential backoff (capped) between retries.
        **kwargs: Additional provider-specific call parameters (for example temperature, max_tokens).
    
    Returns:
        An async generator that yields response chunks as strings.
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

    for attempt in range(max_retries + 1):
        try:
            # Route to appropriate provider
            if use_local:
                async for chunk in local_provider.stream(**call_kwargs):
                    has_yielded = True
                    yield chunk
            else:
                from . import cloud_provider

                async for chunk in cloud_provider.stream(**call_kwargs):
                    has_yielded = True
                    yield chunk
            # If we get here, streaming completed successfully
            return
        except Exception as e:
            last_exception = e

            # If we've already yielded, don't retry
            if has_yielded:
                raise

            # Check if we should retry
            if attempt >= max_retries or not _is_retriable_error(e):
                raise

            # Calculate delay for next attempt
            if exponential_backoff:
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
    api_key: Optional[str] = None,
) -> List[str]:
    """
    Return the list of model identifiers available from the specified provider.
    
    Parameters:
        binding (str): Provider identifier (e.g., "openai", "ollama") used to select provider-specific behavior.
        base_url (str): Provider API base URL to query for models.
        api_key (Optional[str]): Optional API key used for authenticated providers; not required for some local providers.
    
    Returns:
        List[str]: A list of available model names/identifiers.
    """
    if is_local_llm_server(base_url):
        return await local_provider.fetch_models(base_url, api_key)
    else:
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


# API Provider Presets
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
    Return a bundle of provider presets grouped by provider type for frontend display.
    
    Returns:
        ProviderPresetBundle: Mapping with keys "api" and "local" where each value maps provider names to their preset definitions.
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