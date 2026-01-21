# -*- coding: utf-8 -*-
"""
Cloud LLM Provider
==================

Handles all cloud API LLM calls (OpenAI, DeepSeek, Anthropic, etc.)
Provides both complete() and stream() methods.
"""

from collections.abc import AsyncGenerator, Mapping
import logging
import os
import threading
from typing import Any, Protocol, cast

import aiohttp

# Get loggers for suppression during fallback scenarios
# (lightrag logs errors internally before raising exceptions)
_lightrag_logger = logging.getLogger("lightrag")
_openai_logger = logging.getLogger("openai")
logger = logging.getLogger(__name__)

# Thread-safe lock for SSL-warning state
_ssl_warning_lock = threading.Lock()

class OpenAICompleteIfCache(Protocol):
    """Protocol for the lightrag OpenAI completion helper."""

    async def __call__(
        self,
        model: str,
        prompt: str,
        *,
        system_prompt: str,
        history_messages: list[dict[str, str]],
        api_key: str | None,
        base_url: str | None,
        **kwargs: object,
    ) -> str | None:
        """
        Attempt to retrieve a cached completion for the given model and conversation context.
        
        Parameters:
            model (str): Model identifier used to scope the cache lookup.
            prompt (str): User prompt for which a cached completion is sought.
            system_prompt (str): System-level prompt/context to include in the cache key.
            history_messages (list[dict[str, str]]): Prior conversation messages; each message is a mapping with keys like "role" and "content" used to match cached entries.
            api_key (str | None): API key or credential context used to scope or validate the cache lookup.
            base_url (str | None): Provider base URL used to scope the cache lookup.
            **kwargs: Additional provider- or request-specific options that may influence cache matching.
        
        Returns:
            str | None: Cached completion text when a matching entry exists, `None` when no cache hit is available.
        """


# Lazy import for lightrag to avoid import errors when not installed
_openai_complete_if_cache: OpenAICompleteIfCache | None = None


def _get_openai_complete_if_cache() -> OpenAICompleteIfCache:
    """
    Return the module-level OpenAICompleteIfCache callable, loading and caching it from lightrag on first use.
    
    Returns:
        openai_complete_if_cache (OpenAICompleteIfCache): A callable matching the OpenAICompleteIfCache protocol that, when awaited, returns a string completion or `None`.
    """
    global _openai_complete_if_cache
    if _openai_complete_if_cache is None:
        # Import inside the function to avoid circular dependencies
        from lightrag.llm.openai import (  # type: ignore[import-untyped]
            openai_complete_if_cache,
        )

        _openai_complete_if_cache = cast(OpenAICompleteIfCache, openai_complete_if_cache)
    return _openai_complete_if_cache


def _coerce_float(value: object, default: float) -> float:
    """
    Convert a value to a float, returning a fallback when conversion is not appropriate.
    
    Parameters:
        value (object): Input to convert.
        default (float): Fallback value returned when `value` is not a valid numeric input.
    
    Notes:
        Booleans are treated as invalid inputs and cause `default` to be returned (to avoid treating
        `True`/`False` as `1.0`/`0.0`).
    
    Returns:
        float: The converted float, or `default` when conversion is not performed.
    """
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    return default


def _coerce_int(value: object, default: int | None) -> int | None:
    """
    Coerce a value into an integer with a fallback.

    Booleans are rejected to avoid silently treating ``True``/``False`` as
    ``1``/``0``. This mirrors the float coercion behavior and keeps invalid
    inputs from slipping through because ``bool`` is a subclass of ``int``.

    Args:
        value: The raw value.
        default: Value to use when coercion fails.

    Returns:
        An integer value or None.
    """
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    return default


# Use lowercase to avoid constant redefinition warning
_ssl_warning_logged = False


def _get_aiohttp_connector() -> aiohttp.TCPConnector | None:
    """
    Build an optional aiohttp connector with SSL verification disabled.

    Returns:
        A TCPConnector with SSL verification disabled when DISABLE_SSL_VERIFY
        is truthy; otherwise None to use aiohttp defaults.
    """
    # Thread-safe check and one-time warning emission
    disable_flag = os.getenv("DISABLE_SSL_VERIFY", "").lower() in ("true", "1", "yes")
    if not disable_flag:
        return None

    # Emit warning once across threads
    with _ssl_warning_lock:
        if not globals().get("_ssl_warning_logged", False):
            logger.warning(
                "SSL verification is disabled via DISABLE_SSL_VERIFY. This is unsafe and must "
                "not be used in production environments."
            )
            globals()["_ssl_warning_logged"] = True
    return aiohttp.TCPConnector(ssl=False)


from .capabilities import get_effective_temperature, supports_response_format
from .config import get_token_limit_kwargs
from .exceptions import LLMAPIError, LLMAuthenticationError, LLMConfigError
from .utils import (
    build_auth_headers,
    build_chat_url,
    clean_thinking_tags,
    collect_model_names,
    extract_response_content,
    sanitize_url,
)


async def complete(
    prompt: str,
    system_prompt: str = "You are a helpful assistant.",
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    api_version: str | None = None,
    binding: str = "openai",
    **kwargs: Any,
) -> str:
    """
    Perform a single completion request to a cloud LLM provider.
    
    Selects the provider based on `binding` and returns the provider's text response for the given prompt and system context. Supports OpenAI-compatible endpoints (default), Anthropic/Claude, and Cohere; provider-specific parameters such as `max_tokens` and `temperature` may be supplied via `kwargs`.
    
    Parameters:
        prompt (str): The user prompt to complete.
        system_prompt (str): System-level context or instructions for the model.
        model (str | None): Name of the model to use; required.
        api_key (str | None): API key to authenticate with the provider; falls back to environment-based defaults per provider.
        base_url (str | None): Base URL for a custom or self-hosted provider endpoint.
        api_version (str | None): API version to include for Azure/OpenAI-compatible endpoints.
        binding (str): Provider binding identifier (e.g., "openai", "anthropic", "cohere"); case-insensitive.
        **kwargs: Additional provider-specific options (commonly `temperature`, `max_tokens`, and `response_format`).
    
    Returns:
        str: The text completion returned by the selected cloud provider.
    
    Raises:
        LLMConfigError: If `model` is missing or empty.
    """
    binding_lower = (binding or "openai").lower()
    if model is None or not model.strip():
        raise LLMConfigError("Model is required for cloud LLM provider")

    if binding_lower in ["anthropic", "claude"]:
        max_tokens_value = _coerce_int(kwargs.get("max_tokens"), None)
        temperature_value = _coerce_float(kwargs.get("temperature"), 0.7)
        return await _anthropic_complete(
            model=model,
            prompt=prompt,
            system_prompt=system_prompt,
            api_key=api_key,
            base_url=base_url,
            max_tokens=max_tokens_value,
            temperature=temperature_value,
        )

    if binding_lower == "cohere":
        max_tokens_value = _coerce_int(kwargs.get("max_tokens"), None)
        temperature_value = _coerce_float(kwargs.get("temperature"), 0.7)
        return await _cohere_complete(
            model=model,
            prompt=prompt,
            system_prompt=system_prompt,
            api_key=api_key,
            base_url=base_url,
            max_tokens=max_tokens_value,
            temperature=temperature_value,
        )

    # Default to OpenAI-compatible endpoint
    return await _openai_complete(
        model=model,
        prompt=prompt,
        system_prompt=system_prompt,
        api_key=api_key,
        base_url=base_url,
        api_version=api_version,
        binding=binding_lower,
        **kwargs,
    )


async def stream(
    prompt: str,
    system_prompt: str = "You are a helpful assistant.",
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    api_version: str | None = None,
    binding: str = "openai",
    messages: list[dict[str, str]] | None = None,
    **kwargs: Any,
) -> AsyncGenerator[str, None]:
    """
    Stream incremental text chunks from a cloud LLM provider.
    
    Parameters:
        prompt (str): User prompt; ignored if `messages` is provided.
        system_prompt (str): System prompt used when `messages` is not provided.
        model (str | None): Model identifier (required; will raise LLMConfigError if missing).
        api_key (str | None): API key to authenticate the request.
        base_url (str | None): Provider base URL override.
        api_version (str | None): API version (used for Azure/OpenAI-compatible endpoints).
        binding (str): Provider binding (e.g., "openai", "anthropic"); determines provider-specific streaming behavior.
        messages (list[dict[str, str]] | None): Pre-built message list that, if present, overrides `prompt` and `system_prompt`.
        **kwargs: Additional provider-specific options (e.g., `temperature`, `max_tokens`).
    
    Yields:
        str: Successive chunks of generated text from the provider.
    """
    binding_lower = (binding or "openai").lower()
    if model is None or not model.strip():
        raise LLMConfigError("Model is required for cloud LLM provider")

    if binding_lower in ["anthropic", "claude"]:
        max_tokens_value = _coerce_int(kwargs.get("max_tokens"), None)
        temperature_value = _coerce_float(kwargs.get("temperature"), 0.7)
        async for chunk in _anthropic_stream(
            model=model,
            prompt=prompt,
            system_prompt=system_prompt,
            api_key=api_key,
            base_url=base_url,
            messages=messages,
            max_tokens=max_tokens_value,
            temperature=temperature_value,
        ):
            yield chunk
    else:
        async for chunk in _openai_stream(
            model=model,
            prompt=prompt,
            system_prompt=system_prompt,
            api_key=api_key,
            base_url=base_url,
            api_version=api_version,
            binding=binding_lower,
            messages=messages,
            **kwargs,
        ):
            yield chunk


async def _openai_complete(
    model: str,
    prompt: str,
    system_prompt: str,
    api_key: str | None,
    base_url: str | None,
    api_version: str | None = None,
    binding: str = "openai",
    **kwargs: object,
) -> str:
    """
    Obtain a text completion from an OpenAI-compatible cloud LLM, preferring a cached result and falling back to a direct HTTP request.
    
    Attempts to retrieve a cached completion via the lightrag helper; if that fails or returns no content and a base_url is provided, issues a direct POST to the provider's chat endpoint and extracts the first message content. The final returned text has provider-specific "thinking" tags removed.
    
    Parameters:
        model (str): Model identifier to request.
        prompt (str): User prompt to complete.
        system_prompt (str): System/instructional prompt to include.
        api_key (str | None): API key to authenticate the request.
        base_url (str | None): Base URL for the provider's API (used for direct HTTP fallback).
        api_version (str | None): Optional API version (used for Azure-style endpoints).
        binding (str): Provider binding name (e.g., "openai", "azure", "anthropic") affecting URL, headers, and capability handling.
        **kwargs: Provider-specific options (commonly includes `temperature`, `max_tokens` or `max_completion_tokens`, and `response_format`).
    
    Returns:
        str: The cleaned completion text returned by the provider.
    
    Raises:
        LLMAPIError: On network or provider API errors during the direct HTTP fallback.
        LLMConfigError: If no completion could be obtained from cache or direct request.
    """
    # Sanitize URL
    if base_url:
        base_url = sanitize_url(base_url, model)

    # Handle API Parameter Compatibility using capabilities
    # Remove response_format for providers that don't support it (e.g., DeepSeek)
    if not supports_response_format(binding, model):
        kwargs.pop("response_format", None)

    content = None
    try:
        # Try using lightrag's openai_complete_if_cache first (has caching)
        # Only pass api_version if it's set (for Azure OpenAI)
        # Standard OpenAI SDK doesn't accept api_version parameter
        history_messages: list[dict[str, str]] = []
        lightrag_kwargs: dict[str, object] = dict(kwargs)
        lightrag_kwargs.pop("system_prompt", None)
        lightrag_kwargs.pop("history_messages", None)
        lightrag_kwargs.pop("api_key", None)
        lightrag_kwargs.pop("base_url", None)
        lightrag_kwargs.pop("api_version", None)

        # Suppress lightrag's and openai's internal error logging during the call
        # (errors are handled by our fallback mechanism)
        original_lightrag_level = _lightrag_logger.level
        original_openai_level = _openai_logger.level
        _lightrag_logger.setLevel(logging.CRITICAL)
        _openai_logger.setLevel(logging.CRITICAL)
        try:
            # model and prompt must be positional arguments
            if api_version:
                lightrag_kwargs["api_version"] = api_version

            openai_complete_if_cache = _get_openai_complete_if_cache()
            content = await openai_complete_if_cache(
                model,
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                api_key=api_key,
                base_url=base_url,
                **lightrag_kwargs,
            )
        finally:
            _lightrag_logger.setLevel(original_lightrag_level)
            _openai_logger.setLevel(original_openai_level)
    except Exception as exc:
        # Silently ignore lightrag/cache failures to allow fallback to direct aiohttp call
        logger.debug(f"Exception occurred: {exc}")  # Log exception for debugging
    # Fallback: Direct aiohttp call
    if not content and base_url:
        # Build URL using unified utility (use binding for Azure detection)
        url = build_chat_url(base_url, api_version, binding)

        # Build headers using unified utility
        headers = build_auth_headers(api_key, binding)

        data: dict[str, object] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": get_effective_temperature(
                binding, model, _coerce_float(kwargs.get("temperature"), 0.7)
            ),
        }

        # Handle max_tokens / max_completion_tokens based on model
        max_tokens_value = _coerce_int(kwargs.get("max_tokens"), None)
        max_completion_value = _coerce_int(kwargs.get("max_completion_tokens"), None)
        if max_tokens_value is None:
            max_tokens_value = max_completion_value
        if max_tokens_value is None:
            max_tokens_value = 4096
        data.update(get_token_limit_kwargs(model, max_tokens_value))

        # Include response_format if present in kwargs
        response_format = kwargs.get("response_format")
        if response_format is not None:
            data["response_format"] = response_format

        timeout = aiohttp.ClientTimeout(total=120)
        connector = _get_aiohttp_connector()
        async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
            try:
                async with session.post(url, headers=headers, json=data) as resp:
                    if resp.status == 200:
                        result = cast(dict[str, object], await resp.json())
                        choices = result.get("choices")
                        if isinstance(choices, list) and choices:
                            choices_list = cast(list[object], choices)
                            first_choice = choices_list[0]
                            if isinstance(first_choice, Mapping):
                                message = cast(Mapping[str, object], first_choice).get("message")
                            else:
                                message = None
                            if isinstance(message, Mapping):
                                # Use unified response extraction
                                content = extract_response_content(cast(dict[str, object], message))
                    else:
                        error_text = await resp.text()
                        raise LLMAPIError(
                            f"OpenAI API error: {error_text}",
                            status_code=resp.status,
                            provider=binding or "openai",
                        )
            except aiohttp.ClientError as e:
                # Handle connection errors with more specific messages
                if "forcibly closed" in str(e).lower() or "10054" in str(e):
                    raise LLMAPIError(
                        f"Connection to {binding} API was forcibly closed. "
                        "This may indicate network issues or server-side problems. "
                        "Please check your internet connection and try again.",
                        status_code=0,
                        provider=binding or "openai",
                    ) from e
                else:
                    raise LLMAPIError(
                        f"Network error connecting to {binding} API: {e}",
                        status_code=0,
                        provider=binding or "openai",
                    ) from e

    if content is not None:
        # Clean thinking tags from response using unified utility
        return clean_thinking_tags(content, binding, model)

    raise LLMConfigError("Cloud completion failed: no valid configuration")


async def _openai_stream(
    model: str,
    prompt: str,
    system_prompt: str,
    api_key: str | None,
    base_url: str | None,
    api_version: str | None = None,
    binding: str = "openai",
    messages: list[dict[str, str]] | None = None,
    **kwargs: object,
) -> AsyncGenerator[str, None]:
    """
    Stream text chunks from an OpenAI-compatible chat model, yielding cleaned content as it arrives.
    
    Streams delta content from the provider's chunked responses and yields consecutive text fragments. Thinking/agent markers (e.g., <think>…</think>, ◣…◢, 꽁…꽁) are buffered and removed using clean_thinking_tags before yielding; non-JSON or non-data lines are ignored.
    
    Parameters:
        model (str): Model identifier to call.
        prompt (str): User prompt used when `messages` is not provided.
        system_prompt (str): System prompt used when `messages` is not provided.
        api_key (str | None): API key to authenticate the request; may be None to rely on other configuration.
        base_url (str | None): Base URL for the provider; uses the default OpenAI base if None.
        api_version (str | None): Optional API version segment appended to the chat URL.
        binding (str): Provider binding label (e.g., "openai", "anthropic"); used for headers and tag cleaning.
        messages (list[dict[str, str]] | None): Optional explicit messages list; if provided, `prompt`/`system_prompt` are ignored.
        **kwargs: Additional provider parameters; supported keys include `temperature`, `max_tokens` or `max_completion_tokens`, and `response_format`.
    
    Returns:
        AsyncGenerator[str, None]: Streamed text chunks from the model, after cleaning thinking/agent tags.
    
    Raises:
        LLMAPIError: If the HTTP streaming request returns a non-200 status or the provider reports an error.
    """
    import json

    # Sanitize URL
    if base_url:
        base_url = sanitize_url(base_url, model)

    # Handle API Parameter Compatibility using capabilities
    if not supports_response_format(binding, model):
        kwargs.pop("response_format", None)

    # Build URL using unified utility
    effective_base = base_url or "https://api.openai.com/v1"
    url = build_chat_url(effective_base, api_version, binding)

    # Build headers using unified utility
    headers = build_auth_headers(api_key, binding)

    # Build messages
    if messages:
        msg_list = messages
    else:
        msg_list = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

    raw_temp = kwargs.get("temperature")
    temperature = _coerce_float(raw_temp, 0.7)
    data: dict[str, object] = {
        "model": model,
        "messages": msg_list,
        "temperature": temperature,
        "stream": True,
    }

    # Handle max_tokens / max_completion_tokens based on model
    max_tokens_value = _coerce_int(kwargs.get("max_tokens"), None)
    if max_tokens_value is None:
        max_tokens_value = _coerce_int(kwargs.get("max_completion_tokens"), None)
    if max_tokens_value is not None:
        data.update(get_token_limit_kwargs(model, max_tokens_value))

    # Include response_format if present in kwargs
    response_format = kwargs.get("response_format")
    if response_format is not None:
        data["response_format"] = response_format

    timeout = aiohttp.ClientTimeout(total=300)
    connector = _get_aiohttp_connector()
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        async with session.post(url, headers=headers, json=data) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                raise LLMAPIError(
                    f"OpenAI stream error: {error_text}",
                    status_code=resp.status,
                    provider=binding or "openai",
                )

            # Track thinking block state for streaming
            in_thinking_block = False
            thinking_buffer = ""

            async for line in resp.content:
                line_str = line.decode("utf-8").strip()
                if not line_str or not line_str.startswith("data:"):
                    continue

                data_str = line_str[5:].strip()
                if data_str == "[DONE]":
                    break

                try:
                    chunk_data = cast(dict[str, object], json.loads(data_str))
                    choices = chunk_data.get("choices")
                    if isinstance(choices, list) and choices:
                        choices_list = cast(list[object], choices)
                        first_choice = choices_list[0]
                        if isinstance(first_choice, Mapping):
                            delta = cast(Mapping[str, object], first_choice).get("delta")
                        else:
                            delta = None
                        if isinstance(delta, Mapping):
                            content = cast(Mapping[str, object], delta).get("content")
                        else:
                            content = None
                        if isinstance(content, str) and content:
                            # Handle thinking tags in streaming for different marker styles
                            open_markers = ("<think>", "◣", "꽁")
                            close_markers = ("</think>", "◢", "꽁")

                            # Check for start tag (handle split tags)
                            if any(open_m in content for open_m in open_markers):
                                in_thinking_block = True
                                # Handle case where content has text BEFORE <think>
                                for open_m in open_markers:
                                    if open_m in content:
                                        parts = content.split(open_m, 1)
                                        if parts[0]:
                                            yield parts[0]
                                        thinking_buffer = open_m + parts[1]

                                        # Check if closed immediately in same chunk
                                        if any(close_m in thinking_buffer for close_m in close_markers):
                                            cleaned = clean_thinking_tags(thinking_buffer, binding, model)
                                            if cleaned:
                                                yield cleaned
                                            thinking_buffer = ""
                                            in_thinking_block = False
                                        break
                                continue
                            elif in_thinking_block:
                                thinking_buffer += content
                                if any(close_m in thinking_buffer for close_m in close_markers):
                                    # Block finished
                                    cleaned = clean_thinking_tags(thinking_buffer, binding, model)
                                    if cleaned:
                                        yield cleaned
                                    in_thinking_block = False
                                    thinking_buffer = ""
                                continue
                            else:
                                yield content
                except json.JSONDecodeError:
                    continue


async def _anthropic_complete(
    model: str,
    prompt: str,
    system_prompt: str,
    api_key: str | None,
    base_url: str | None,
    messages: list[dict[str, str]] | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
) -> str:
    """
    Perform a completion using the Anthropic (Claude) API.
    
    Parameters:
        model (str): Anthropic model identifier to use.
        prompt (str): User prompt text used when `messages` is not provided.
        system_prompt (str): System-level instruction sent as the Anthropic `system` field.
        api_key (str | None): Anthropic API key; if None, the `ANTHROPIC_API_KEY` environment variable is used.
        base_url (str | None): Base URL for the Anthropic API; defaults to "https://api.anthropic.com/v1" when None.
        messages (list[dict[str, str]] | None): Optional pre-built message list. Each message should be a mapping with keys `"role"` and `"content"`. Any message with `"role" == 'system'` will be extracted and used as `system_prompt`; other messages are sent as the Anthropic `messages` payload.
        max_tokens (int | None): Maximum tokens for the completion; defaults to 4096 when None.
        temperature (float | None): Sampling temperature; defaults to 0.7 when None.
    
    Returns:
        str: The text content of the first completion item returned by the Anthropic API.
    
    Raises:
        LLMAuthenticationError: If no API key is provided via argument or environment.
        LLMAPIError: If the Anthropic API responds with a non-200 status or an unexpected payload.
    """
    api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise LLMAuthenticationError("Anthropic API key is missing.", provider="anthropic")

    # Build URL using unified utility
    effective_base = base_url or "https://api.anthropic.com/v1"
    url = build_chat_url(effective_base, binding="anthropic")

    # Build headers using unified utility
    headers = build_auth_headers(api_key, binding="anthropic")

    # Build messages - handle pre-built messages array
    if messages:
        # Filter out system messages for Anthropic (system is a separate parameter)
        msg_list = [m for m in messages if m.get("role") != "system"]
        system_content = next(
            (m["content"] for m in messages if m.get("role") == "system"),
            system_prompt,
        )
    else:
        msg_list = [{"role": "user", "content": prompt}]
        system_content = system_prompt

    max_tokens_value = max_tokens if max_tokens is not None else 4096
    temperature_value = temperature if temperature is not None else 0.7
    data: dict[str, object] = {
        "model": model,
        "system": system_content,
        "messages": msg_list,
        "max_tokens": max_tokens_value,
        "temperature": temperature_value,
    }

    timeout = aiohttp.ClientTimeout(total=120)
    connector = _get_aiohttp_connector()
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        async with session.post(url, headers=headers, json=data) as response:
            if response.status != 200:
                error_text = await response.text()
                raise LLMAPIError(
                    f"Anthropic API error: {error_text}",
                    status_code=response.status,
                    provider="anthropic",
                )

            result = cast(dict[str, object], await response.json())
            content_items = result.get("content")
            if isinstance(content_items, list) and content_items:
                content_list = cast(list[object], content_items)
                first_item = content_list[0]
                if isinstance(first_item, Mapping):
                    text = cast(Mapping[str, object], first_item).get("text")
                    if isinstance(text, str):
                        return text
            raise LLMAPIError(
                "Anthropic API error: unexpected response payload",
                status_code=response.status,
                provider="anthropic",
            )


async def _anthropic_stream(
    model: str,
    prompt: str,
    system_prompt: str,
    api_key: str | None,
    base_url: str | None,
    messages: list[dict[str, str]] | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
) -> AsyncGenerator[str, None]:
    """
    Stream response text from Anthropic (Claude) for a chat request.
    
    Messages provided by `messages` will be sent as the conversation (system messages are removed for Anthropic);
    if a system message is present its content takes precedence over `system_prompt`. If `messages` is not provided,
    the `prompt` is used as the single user message and `system_prompt` is used as the system content.
    Defaults: `max_tokens` = 4096, `temperature` = 0.7. The `api_key` falls back to the `ANTHROPIC_API_KEY` environment variable if not set.
    
    Parameters:
        model (str): Anthropic model identifier.
        prompt (str): User prompt used when `messages` is not provided.
        system_prompt (str): System prompt used when no system message is present in `messages`.
        api_key (str | None): Anthropic API key or None to read from `ANTHROPIC_API_KEY`.
        base_url (str | None): Base URL for the Anthropic API; defaults to Anthropic's public endpoint when None.
        messages (list[dict[str, str]] | None): Optional pre-built message list; system messages are filtered out for the API.
        max_tokens (int | None): Maximum tokens to generate; defaults to 4096 when None.
        temperature (float | None): Sampling temperature; defaults to 0.7 when None.
    
    Returns:
        AsyncGenerator[str, None]: An async generator yielding response text chunks as they arrive from the stream.
    
    Raises:
        LLMAuthenticationError: If no API key is available.
        LLMAPIError: If the Anthropic API responds with a non-200 status.
    """
    import json

    api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise LLMAuthenticationError("Anthropic API key is missing.", provider="anthropic")

    # Build URL using unified utility
    effective_base = base_url or "https://api.anthropic.com/v1"
    url = build_chat_url(effective_base, binding="anthropic")

    # Build headers using unified utility
    headers = build_auth_headers(api_key, binding="anthropic")

    # Build messages
    if messages:
        # Filter out system messages for Anthropic
        msg_list = [m for m in messages if m.get("role") != "system"]
        system_content = next(
            (m["content"] for m in messages if m.get("role") == "system"),
            system_prompt,
        )
    else:
        msg_list = [{"role": "user", "content": prompt}]
        system_content = system_prompt

    max_tokens_value = max_tokens if max_tokens is not None else 4096
    temperature_value = temperature if temperature is not None else 0.7
    data: dict[str, object] = {
        "model": model,
        "system": system_content,
        "messages": msg_list,
        "max_tokens": max_tokens_value,
        "temperature": temperature_value,
        "stream": True,
    }

    timeout = aiohttp.ClientTimeout(total=300)
    connector = _get_aiohttp_connector()
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        async with session.post(url, headers=headers, json=data) as response:
            if response.status != 200:
                error_text = await response.text()
                raise LLMAPIError(
                    f"Anthropic stream error: {error_text}",
                    status_code=response.status,
                    provider="anthropic",
                )

            async for line in response.content:
                line_str = line.decode("utf-8").strip()
                if not line_str or not line_str.startswith("data:"):
                    continue

                data_str = line_str[5:].strip()
                if not data_str:
                    continue

                try:
                    chunk_data = cast(dict[str, object], json.loads(data_str))
                    event_type = chunk_data.get("type")
                    if event_type == "content_block_delta":
                        delta = chunk_data.get("delta")
                        if isinstance(delta, Mapping):
                            text = cast(Mapping[str, object], delta).get("text")
                        else:
                            text = None
                        if isinstance(text, str) and text:
                            yield text
                except json.JSONDecodeError:
                    continue


async def _cohere_complete(
    model: str,
    prompt: str,
    system_prompt: str,
    api_key: str | None,
    base_url: str | None,
    max_tokens: int | None = None,
    temperature: float | None = None,
) -> str:
    """
    Request a completion from the Cohere chat API for the specified model and prompts.
    
    Parameters:
        api_key (str | None): Cohere API key to use; if None, the `COHERE_API_KEY` environment variable is used.
        base_url (str | None): Base URL for the Cohere API; defaults to "https://api.cohere.ai/v1" when not provided.
        max_tokens (int | None): Maximum number of tokens to generate; defaults to 4096 when not provided.
        temperature (float | None): Sampling temperature for generation; defaults to 0.7 when not provided.
    
    Returns:
        str: The text content returned by Cohere.
    
    Raises:
        LLMAuthenticationError: If no API key is available.
        LLMAPIError: If the Cohere API returns a non-200 status or an unexpected payload.
    """
    api_key = api_key or os.getenv("COHERE_API_KEY")
    if not api_key:
        raise LLMAuthenticationError("Cohere API key is missing.", provider="cohere")

    # Build URL using unified utility
    effective_base = base_url or "https://api.cohere.ai/v1"
    url = f"{effective_base}/chat"

    # Build headers using unified utility
    headers = build_auth_headers(api_key, binding="cohere")

    max_tokens_value = max_tokens if max_tokens is not None else 4096
    temperature_value = temperature if temperature is not None else 0.7
    data: dict[str, object] = {
        "model": model,
        "message": f"{system_prompt}\n\n{prompt}",
        "max_tokens": max_tokens_value,
        "temperature": temperature_value,
    }

    timeout = aiohttp.ClientTimeout(total=120)
    connector = _get_aiohttp_connector()
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        async with session.post(url, headers=headers, json=data) as response:
            if response.status != 200:
                error_text = await response.text()
                raise LLMAPIError(
                    f"Cohere API error: {error_text}",
                    status_code=response.status,
                    provider="cohere",
                )

            result = cast(dict[str, object], await response.json())
            text = result.get("text")
            if isinstance(text, str):
                return text
            raise LLMAPIError(
                "Cohere API error: unexpected response payload",
                status_code=response.status,
                provider="cohere",
            )


async def fetch_models(
    base_url: str,
    api_key: str | None = None,
    binding: str = "openai",
) -> list[str]:
    """
    Fetch available model identifiers from the given cloud LLM provider endpoint.
    
    Parameters:
        base_url (str): Base API URL (may include path; trailing slash is permitted).
        api_key (str | None): Provider API key to include in authorization headers.
        binding (str): Provider type identifier (e.g., "openai", "anthropic"); compared case-insensitively.
    
    Returns:
        list[str]: A list of available model names/identifiers; empty list on error or if none found.
    """
    binding = binding.lower()
    base_url = base_url.rstrip("/")

    # Build headers using unified utility
    headers = build_auth_headers(api_key, binding)
    # Remove Content-Type for GET request
    headers.pop("Content-Type", None)

    timeout = aiohttp.ClientTimeout(total=30)
    connector = _get_aiohttp_connector()
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        try:
            url = f"{base_url}/models"
            async with session.get(url, headers=headers) as resp:
                if resp.status == 200:
                    payload = await resp.json()
                    if isinstance(payload, Mapping):
                        mapping = cast(Mapping[str, object], payload)
                        items = mapping.get("data")
                        if isinstance(items, list):
                            return collect_model_names(cast(list[object], items))
                    elif isinstance(payload, list):
                        return collect_model_names(cast(list[object], payload))
            return []
        except Exception as e:
            logger.error("Error fetching models from %s: %s", base_url, e)
            return []


__all__ = [
    "complete",
    "stream",
    "fetch_models",
]