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
from types import SimpleNamespace

import aiohttp

# Get loggers for suppression during fallback scenarios
# (lightrag logs errors internally before raising exceptions)
_lightrag_logger = logging.getLogger("lightrag")
_openai_logger = logging.getLogger("openai")
logger = logging.getLogger(__name__)

# Thread-safe lock for SSL-warning state
_ssl_warning_lock = threading.Lock()
# One-time flag to ensure SSL-disable warning is logged once
_ssl_warning_logged: bool = False


def run_tls_diagnostics(target_url: str) -> dict[str, object]:
    """Run lightweight TLS diagnostics against the host:port extracted from URL.

    Returns a small dictionary containing outcome information to help debug
    transient TLS/SSL failures (protocol, cipher, cert subject, error).
    """
    from urllib.parse import urlparse

    parsed = urlparse(target_url)
    host = parsed.hostname or ""
    port = parsed.port or (443 if parsed.scheme in ("https", "wss") else 80)
    info: dict[str, object] = {"host": host, "port": port}

    try:
        # Quick TCP connect
        with socket.create_connection((host, port), timeout=5) as sock:
            try:
                ctx = ssl.create_default_context()
                with ctx.wrap_socket(sock, server_hostname=host) as ssock:
                    info["peer_cert_subject"] = ssock.getpeercert().get("subject")
                    info["cipher"] = ssock.cipher()
                    # ssl.SSLSocket has version() returning protocol
                    info["protocol"] = getattr(ssock, "version", lambda: None)()
                    info["success"] = True
            except Exception as tls_exc:
                info["tls_error"] = str(tls_exc)
                info["success"] = False
    except Exception as conn_exc:
        info["connect_error"] = str(conn_exc)
        info["success"] = False

    # Environment hints
    info["https_proxy"] = os.getenv("HTTPS_PROXY") or os.getenv("https_proxy")
    info["http_proxy"] = os.getenv("HTTP_PROXY") or os.getenv("http_proxy")
    info["disable_ssl_verify"] = os.getenv("DISABLE_SSL_VERIFY")
    return info


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


def _get_aiohttp_connector() -> aiohttp.TCPConnector | None:
    global _ssl_warning_logged
    # Thread-safe check and one-time warning emission
    disable_flag = os.getenv("DISABLE_SSL_VERIFY", "").lower() in ("true", "1", "yes")
    if not disable_flag:
        return None

    # Emit warning once across threads
    with _ssl_warning_lock:
        if not _ssl_warning_logged:
            logger.warning(
                "SSL verification is disabled via DISABLE_SSL_VERIFY. This is unsafe and must "
                "not be used in production environments."
            )
            _ssl_warning_logged = True
    return aiohttp.TCPConnector(ssl=False)


import socket
import ssl

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
    binding_lower = (binding or "openai").lower()
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
        if binding_lower not in ["azure", "azure_openai"]:
            from src.logging.logger import suppressed_logging

            # model and prompt must be positional arguments
            lightrag_kwargs["system_prompt"] = system_prompt
            lightrag_kwargs["history_messages"] = history_messages
            lightrag_kwargs["api_key"] = api_key
            if base_url:
                lightrag_kwargs["base_url"] = base_url
            if api_version:
                lightrag_kwargs["api_version"] = api_version

            # Use a filter-based suppression to avoid race conditions with log levels
            with suppressed_logging(["lightrag", "openai"], level=logging.CRITICAL):
                try:
                    content = await openai_complete_if_cache(model, prompt, **lightrag_kwargs)
                except Exception:
                    # Swallow errors - we'll fall back to the standard cloud call
                    content = None
    except Exception:
        # Log the exception with stack info for debuggability and continue to fallback
        logger.debug("Optional dependency lightrag unavailable", exc_info=True)
    # Fallback: Direct aiohttp call
    if not content:
        effective_base = base_url or "https://api.openai.com/v1"
        # Build URL using unified utility (use binding for Azure detection)
        url = build_chat_url(effective_base, api_version, binding, model)


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
                # Enhanced diagnostics for TLS/SSL failures and network errors.
                err_str = str(e).lower()

                # If this appears to be SSL-related, run TLS diagnostics to gather context
                is_ssl_error = isinstance(e, ssl.SSLError) or "ssl" in err_str or "sslv3" in err_str
                diagnostic_info = None
                if is_ssl_error:
                    try:
                        diagnostic_info = run_tls_diagnostics(url)
                        logger.warning(f"TLS diagnostic for {url}: {diagnostic_info}")
                    except Exception as diag_exc:
                        logger.debug(f"TLS diagnostics failed: {diag_exc}", exc_info=True)

                if "forcibly closed" in err_str or "10054" in err_str:
                    raise LLMAPIError(
                        f"Connection to {binding} API was forcibly closed. "
                        "This may indicate network issues or server-side problems. "
                        "Please check your internet connection and try again.",
                        status_code=0,
                        provider=binding or "openai",
                    ) from e

                # Include diagnostic summary in the error message if available
                if diagnostic_info:
                    raise LLMAPIError(
                        f"Network/SSL error connecting to {binding} API: {e}; diagnostics: {diagnostic_info}",
                        status_code=0,
                        provider=binding or "openai",
                    ) from e

                raise LLMAPIError(
                    f"Network error connecting to {binding} API: {e}",
                    status_code=0,
                    provider=binding or "openai",
                ) from e

    if isinstance(content, str):
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
    url = build_chat_url(effective_base, api_version, binding, model)

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
            _stream_context = SimpleNamespace(
                yield_buffer="",
                thinking_buffer="",
                in_thinking_block=False,
            )

            async for line in resp.content:
                line_str = line.decode("utf-8").strip()
                if not line_str or not line_str.startswith("data:"):
                    continue

                data_str = line_str[5:].strip()
                if data_str == "[DONE]":
                    break

                try:
                    chunk_data = cast(dict[str, object], json.loads(data_str))
                except json.JSONDecodeError:
                    # Malformed JSON frame, skip this SSE data
                    logger.warning("Skipping malformed SSE JSON chunk: %.100s", data_str)
                    continue

                # Defensive guards for unusual provider frames
                choices = chunk_data.get("choices")
                if not (isinstance(choices, list) and choices):
                    continue

                first_choice = choices[0]
                if not isinstance(first_choice, Mapping):
                    continue

                delta = cast(Mapping[str, object], first_choice).get("delta", {})
                if not isinstance(delta, Mapping):
                    continue

                content = delta.get("content")
                if not isinstance(content, str) or not content:
                    continue

                # Use shared StreamParser to handle buffering and marker logic
                parser = getattr(_stream_context, "parser", None)
                if parser is None:
                    from src.services.llm.stream_parser import StreamParser

                    parser = StreamParser(binding=binding, model=model)
                    _stream_context.parser = parser

                for out in parser.append(content):
                    yield out

            # End of streaming loop - flush any remaining buffered content
            parser = getattr(_stream_context, "parser", None)
            if parser is not None:
                for out in parser.finalize():
                    yield out


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
