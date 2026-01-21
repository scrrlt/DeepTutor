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
        """Return cached completion content when available."""


# Lazy import for lightrag to avoid import errors when not installed
_openai_complete_if_cache: OpenAICompleteIfCache | None = None


def _get_openai_complete_if_cache() -> OpenAICompleteIfCache:
    """Lazy load openai_complete_if_cache from lightrag."""
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
    Coerce a value into a float with a fallback.

    Booleans are treated specially because ``bool`` is a subclass of ``int`` in
    Python. Coercing ``True``/``False`` into ``1.0``/``0.0`` would hide invalid
    inputs, so we fall back to the default instead.

    Args:
        value: The raw value.
        default: Value to use when coercion fails.

    Returns:
        A float value.
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
        global _ssl_warning_logged
        if not _ssl_warning_logged:
            logger.warning(
                "SSL verification is disabled via DISABLE_SSL_VERIFY. This is unsafe and must "
                "not be used in production environments."
            )
            _ssl_warning_logged = True
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
    Complete a prompt using cloud API providers.

    Supports OpenAI-compatible APIs and Anthropic.

    Args:
        prompt: The user prompt
        system_prompt: System prompt for context
        model: Model name
        api_key: API key
        base_url: Base URL for the API
        api_version: API version for Azure OpenAI
        binding: Provider binding type (openai, anthropic)
        **kwargs: Additional parameters (temperature, max_tokens, etc.)

    Returns:
        str: The LLM response
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
    Stream a response from cloud API providers.

    Args:
        prompt: The user prompt (ignored if messages provided)
        system_prompt: System prompt for context
        model: Model name
        api_key: API key
        base_url: Base URL for the API
        api_version: API version for Azure OpenAI
        binding: Provider binding type (openai, anthropic)
        messages: Pre-built messages array (optional, overrides prompt/system_prompt)
        **kwargs: Additional parameters (temperature, max_tokens, etc.)

    Yields:
        str: Response chunks
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
    """OpenAI-compatible completion."""
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

        # Try using lightrag's cached complete implementation when available.
        # We avoid mutating logger levels across an await point because that is
        # unsafe in asyncio (other tasks may run concurrently and observe
        # intermediate global level changes). Instead, use a temporary filter
        # that suppresses non-critical messages for the named loggers.
        openai_complete_if_cache = _get_openai_complete_if_cache()
        if openai_complete_if_cache is not None:
            from src.logging.logger import suppressed_logging

            # model and prompt must be positional arguments
            if api_version:
                lightrag_kwargs["api_version"] = api_version

            # Use a filter-based suppression to avoid race conditions with log levels
            with suppressed_logging(["lightrag", "openai"], level=logging.CRITICAL):
                try:
                    content = await openai_complete_if_cache(
                        model,
                        prompt,
                        system_prompt=system_prompt,
                        history_messages=history_messages,
                        api_key=api_key,
                    )
                except Exception:
                    # Swallow errors - we'll fall back to the standard cloud call
                    content = None
        else:
            # lightrag caching not available; proceed with normal cloud call
            content = None
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
    """OpenAI-compatible streaming."""
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

            # Per-stream context object to hold partial buffers
            from types import SimpleNamespace

            _stream_context = SimpleNamespace(
                yield_buffer="",
                thinking_buffer="",
                in_thinking_block=False,
            )

            def _ends_with_partial(buf: str, markers: tuple[str, ...]) -> bool:
                for m in markers:
                    for k in range(1, len(m)):
                        if buf.endswith(m[:k]):
                            return True
                return False

            def _find_first_marker(buf: str, markers: tuple[str, ...]):
                first_idx = None
                first_marker = None
                for m in markers:
                    idx = buf.find(m)
                    if idx != -1 and (first_idx is None or idx < first_idx):
                        first_idx = idx
                        first_marker = m
                return first_idx, first_marker

            def _process_buffers(open_markers: tuple[str, ...], close_markers: tuple[str, ...], binding: str | None = None, model: str | None = None):
                """Process buffers in _stream_context and yield ready outputs."""
                while True:
                    yield_buffer = _stream_context.yield_buffer
                    thinking_buffer = _stream_context.thinking_buffer
                    in_thinking_block = _stream_context.in_thinking_block

                    progressed = False

                    if not in_thinking_block:
                        idx, marker = _find_first_marker(yield_buffer, open_markers)

                        # If buffer ends with a partial open marker, wait for more
                        if _ends_with_partial(yield_buffer, open_markers):
                            break

                        if idx is not None:
                            # Yield text before the marker
                            if idx > 0:
                                pre = yield_buffer[:idx]
                                if pre:
                                    yield pre
                            # Move rest into thinking buffer
                            thinking_buffer = yield_buffer[idx:]
                            yield_buffer = ""
                            in_thinking_block = True

                            # If closed already in same buffer, process immediately
                            close_idx, close_marker = _find_first_marker(thinking_buffer, close_markers)
                            if close_idx is not None:
                                after = thinking_buffer.split(close_marker, 1)[1]
                                cleaned = clean_thinking_tags(thinking_buffer, binding, model)
                                if cleaned:
                                    yield cleaned
                                thinking_buffer = ""
                                in_thinking_block = False
                                yield_buffer = after
                                progressed = True
                                _stream_context.yield_buffer = yield_buffer
                                _stream_context.thinking_buffer = thinking_buffer
                                _stream_context.in_thinking_block = in_thinking_block
                                continue
                        else:
                            # No marker, yield whole buffer
                            if yield_buffer:
                                yield yield_buffer
                                yield_buffer = ""
                    else:
                        # inside thinking block
                        thinking_buffer += yield_buffer
                        yield_buffer = ""

                        # If ends with partial close marker, wait for more
                        if _ends_with_partial(thinking_buffer, close_markers):
                            break

                        close_idx, close_marker = _find_first_marker(thinking_buffer, close_markers)
                        if close_idx is not None:
                            after = thinking_buffer.split(close_marker, 1)[1]
                            cleaned = clean_thinking_tags(thinking_buffer, binding, model)
                            if cleaned:
                                yield cleaned
                            thinking_buffer = ""
                            in_thinking_block = False
                            yield_buffer = after
                            progressed = True
                            _stream_context.yield_buffer = yield_buffer
                            _stream_context.thinking_buffer = thinking_buffer
                            _stream_context.in_thinking_block = in_thinking_block
                            continue
                        # otherwise, still inside thinking block and no close yet
                        break

                    _stream_context.yield_buffer = yield_buffer
                    _stream_context.thinking_buffer = thinking_buffer
                    _stream_context.in_thinking_block = in_thinking_block

                    if not progressed:
                        break

            async for line in resp.content:
                line_str = line.decode("utf-8").strip()
                if not line_str or not line_str.startswith("data:"):
                    continue

                data_str = line_str[5:].strip()
                if data_str == "[DONE]":
                    break

            # End of streaming loop - ensure any remaining buffers are emitted
            if "_stream_context" in locals():
                yield_buffer = getattr(_stream_context, "yield_buffer", "")
                thinking_buffer = getattr(_stream_context, "thinking_buffer", "")
                in_thinking_block = getattr(_stream_context, "in_thinking_block", False)

                if in_thinking_block and thinking_buffer:
                    cleaned = clean_thinking_tags(thinking_buffer, binding, model)
                    if cleaned:
                        yield cleaned
                elif yield_buffer:
                    yield yield_buffer

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

                            # Append content and let the shared processor handle it
                            _stream_context.yield_buffer += content

                            for out in _process_buffers(open_markers, close_markers, binding, model):
                                yield out                                            yield_buffer = ""
                                else:
                                    # inside thinking block
                                    thinking_buffer += yield_buffer
                                    yield_buffer = ""

                                    # If ends with partial close marker, wait for more
                                    if _ends_with_partial(thinking_buffer, close_markers):
                                        break

                                    close_idx, close_marker = _find_first_marker(thinking_buffer, close_markers)
                                    if close_idx is not None:
                                        after = thinking_buffer.split(close_marker, 1)[1]
                                        cleaned = clean_thinking_tags(thinking_buffer, binding, model)
                                        if cleaned:
                                            yield cleaned
                                        thinking_buffer = ""
                                        in_thinking_block = False
                                        yield_buffer = after
                                        processed = True
                                        continue
                                    # otherwise, still inside thinking block and no close yet
                                    break

                            # Update context
                            _stream_context.yield_buffer = yield_buffer
                            _stream_context.thinking_buffer = thinking_buffer
                            _stream_context.in_thinking_block = in_thinking_block

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
    """Anthropic (Claude) API completion."""
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
    """Anthropic (Claude) API streaming."""
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
    """Cohere API completion."""
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
    Fetch available models from cloud provider.

    Args:
        base_url: API endpoint URL
        api_key: API key
        binding: Provider type (openai, anthropic)

    Returns:
        List of available model names
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
