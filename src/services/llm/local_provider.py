# -*- coding: utf-8 -*-
"""
Local LLM Provider
==================

Handles all local/self-hosted LLM calls (LM Studio, Ollama, vLLM, llama.cpp, etc.)
Uses aiohttp instead of httpx for better compatibility with local servers.

Key features:
- Uses aiohttp (httpx has known 502 issues with some local servers like LM Studio)
- Handles thinking tags (<think>) from reasoning models like Qwen
- Extended timeouts for potentially slower local inference
"""

from collections.abc import AsyncGenerator
from typing import Any

import json
import logging
import os
import threading

import aiohttp

from src.services.llm.stream_parser import StreamParser

from .exceptions import LLMAPIError, LLMConfigError
from .utils import (
    build_auth_headers,
    build_chat_url,
    clean_thinking_tags,
    collect_model_names,
    extract_response_content,
    sanitize_url,
)

# ruff: noqa: TRY003
logger = logging.getLogger(__name__)

# Extended timeout for local servers (may be slower than cloud)
DEFAULT_TIMEOUT = 300  # 5 minutes

_ssl_warning_lock = threading.Lock()
_ssl_warning_logged = False


def _get_aiohttp_connector() -> aiohttp.TCPConnector | None:
    """Create an aiohttp connector with optional SSL verification disabled.

    Returns:
        TCPConnector when SSL is disabled, otherwise None.

    Raises:
        None.
    """
    global _ssl_warning_logged
    disable_flag = os.getenv("DISABLE_SSL_VERIFY", "").lower() in ("true", "1", "yes")
    if not disable_flag:
        return None

    with _ssl_warning_lock:
        if not _ssl_warning_logged:
            logger.warning(
                "SSL verification is disabled via DISABLE_SSL_VERIFY for local LLM. "
                "This is unsafe."
            )
            _ssl_warning_logged = True
    return aiohttp.TCPConnector(ssl=False)


async def _iter_stream_lines(
    stream: aiohttp.StreamReader,
) -> AsyncGenerator[str, None]:
    """Yield decoded lines from a stream reader.

    Args:
        stream: Aiohttp stream reader for the response body.

    Yields:
        Decoded lines with CRLF stripped.

    Raises:
        None.
    """
    buffer = ""
    async for chunk in stream:
        buffer += chunk.decode("utf-8", errors="ignore")
        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            yield line.rstrip("\r")
    if buffer:
        yield buffer.rstrip("\r")


async def complete(
    prompt: str,
    system_prompt: str = "You are a helpful assistant.",
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    messages: list[dict[str, str]] | None = None,
    **kwargs: Any,
) -> str:
    """
    Complete a prompt using local LLM server.

    Uses aiohttp for better compatibility with local servers.

    Args:
        prompt: The user prompt (ignored if messages provided)
        system_prompt: System prompt for context
        model: Model name
        api_key: API key (optional for most local servers)
        base_url: Base URL for the local server
        messages: Pre-built messages array (optional)
        **kwargs: Additional parameters (temperature, max_tokens, etc.)

    Returns:
        str: The LLM response
    """
    if not base_url:
        raise LLMConfigError("base_url is required for local LLM provider")  # noqa: TRY003

    # Sanitize URL and build chat endpoint
    base_url = sanitize_url(base_url, model or "")
    url = build_chat_url(base_url)

    # Build headers using unified utility
    headers = build_auth_headers(api_key)

    # Build messages
    if messages:
        msg_list = messages
    else:
        msg_list = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

    # Build request data
    data = {
        "model": model or "default",
        "messages": msg_list,
        "temperature": kwargs.get("temperature", 0.7),
        "stream": False,
    }

    # Add optional parameters
    if kwargs.get("max_tokens"):
        data["max_tokens"] = kwargs["max_tokens"]

    timeout = aiohttp.ClientTimeout(total=kwargs.get("timeout", DEFAULT_TIMEOUT))

    connector = _get_aiohttp_connector()

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        async with session.post(url, json=data, headers=headers) as response:
            if response.status != 200:
                error_text = await response.text()
                logger.error("Local LLM error: %s", error_text[:200])
                raise LLMAPIError(
                    "Local LLM error",
                    status_code=response.status,
                    provider="local",
                )  # noqa: TRY003
            result = await response.json()

            if "choices" in result and result["choices"]:
                msg = result["choices"][0].get("message", {})
                # Use unified response extraction
                content = extract_response_content(msg)
                # Clean thinking tags using unified utility (pass binding/model)
                content = clean_thinking_tags(content, "local", model)
                if content:
                    return content

            logger.warning("Local LLM returned no choices: %s", result)
            return ""


async def stream(
    prompt: str,
    system_prompt: str = "You are a helpful assistant.",
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    messages: list[dict[str, str]] | None = None,
    **kwargs: Any,
) -> AsyncGenerator[str, None]:
    """
    Stream a response from local LLM server.

    Uses aiohttp for better compatibility with local servers.
    Falls back to non-streaming if streaming fails.

    Args:
        prompt: The user prompt (ignored if messages provided)
        system_prompt: System prompt for context
        model: Model name
        api_key: API key (optional for most local servers)
        base_url: Base URL for the local server
        messages: Pre-built messages array (optional)
        **kwargs: Additional parameters (temperature, max_tokens, etc.)

    Yields:
        str: Response chunks
    """
    if not base_url:
        raise LLMConfigError("base_url is required for local LLM provider")  # noqa: TRY003

    # Sanitize URL and build chat endpoint
    base_url = sanitize_url(base_url, model or "")
    url = build_chat_url(base_url)

    # Build headers using unified utility
    headers = build_auth_headers(api_key)

    # Build messages
    if messages:
        msg_list = messages
    else:
        msg_list = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

    # Build request data
    data = {
        "model": model or "default",
        "messages": msg_list,
        "temperature": kwargs.get("temperature", 0.7),
        "stream": True,
    }

    if kwargs.get("max_tokens"):
        data["max_tokens"] = kwargs["max_tokens"]

    timeout = aiohttp.ClientTimeout(total=kwargs.get("timeout", DEFAULT_TIMEOUT))

    try:
        connector = _get_aiohttp_connector()
        async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
            async with session.post(url, json=data, headers=headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error("Local LLM stream error: %s", error_text[:200])
                    raise LLMAPIError(
                        "Local LLM error",
                        status_code=response.status,
                        provider="local",
                    )  # noqa: TRY003
                parser: StreamParser | None = None
                data_lines: list[str] = []

                def _emit_content(text: str) -> None:
                    nonlocal parser
                    if not text:
                        return
                    if parser is None:
                        parser = StreamParser(binding="local", model=model)
                    for out in parser.append(text):
                        yield out

                async for line_str in _iter_stream_lines(response.content):
                    if line_str == "":
                        if data_lines:
                            data_str = "\n".join(data_lines).strip()
                            data_lines = []
                            if data_str == "[DONE]":
                                break
                            try:
                                chunk_data = json.loads(data_str)
                            except json.JSONDecodeError:
                                logger.warning(
                                    "Skipping malformed SSE JSON chunk: %s...",
                                    data_str[:100],
                                )
                                continue
                            choices = chunk_data.get("choices")
                            if isinstance(choices, list) and choices:
                                delta = choices[0].get("delta", {})
                                content = delta.get("content")
                                if isinstance(content, str):
                                    for out in _emit_content(content):
                                        yield out
                        continue

                    if line_str.startswith(":") or line_str.startswith("event:"):
                        continue
                    if line_str.startswith("id:"):
                        continue
                    if line_str.startswith("data:"):
                        data_lines.append(line_str[5:].lstrip())
                        continue
                    if line_str.startswith("{"):
                        try:
                            chunk_data = json.loads(line_str)
                        except json.JSONDecodeError:
                            logger.warning("Skipping malformed JSON chunk: %s...", line_str[:100])
                            continue
                        choices = chunk_data.get("choices")
                        if isinstance(choices, list) and choices:
                            delta = choices[0].get("delta", {})
                            content = delta.get("content")
                            if isinstance(content, str):
                                for out in _emit_content(content):
                                    yield out

                if data_lines:
                    data_str = "\n".join(data_lines).strip()
                    if data_str and data_str != "[DONE]":
                        try:
                            chunk_data = json.loads(data_str)
                            choices = chunk_data.get("choices")
                            if isinstance(choices, list) and choices:
                                delta = choices[0].get("delta", {})
                                content = delta.get("content")
                                if isinstance(content, str):
                                    for out in _emit_content(content):
                                        yield out
                        except json.JSONDecodeError:
                            logger.warning(
                                "Skipping malformed SSE JSON chunk at stream end: %s...",
                                data_str[:100],
                            )

                if parser is not None:
                    for out in parser.finalize():
                        yield out

    except LLMAPIError:
        raise  # Re-raise LLM errors as-is
    except Exception as e:
        # Streaming failed, fall back to non-streaming
        logger.warning("Streaming failed (%s), falling back to non-streaming", e)

        try:
            content = await complete(
                prompt=prompt,
                system_prompt=system_prompt,
                model=model,
                api_key=api_key,
                base_url=base_url,
                messages=messages,
                **kwargs,
            )
            if content:
                yield content
        except Exception as e2:
            logger.error("Local LLM failed: streaming=%s, non-streaming=%s", e, e2)
            raise LLMAPIError(
                "Local LLM failed",
            )  # noqa: TRY003


async def fetch_models(
    base_url: str,
    api_key: str | None = None,
) -> list[str]:
    """
    Fetch available models from local LLM server.

    Supports:
    - Ollama (/api/tags)
    - OpenAI-compatible (/models)

    Args:
        base_url: Base URL for the local server
        api_key: API key (optional)

    Returns:
        List of available model names
    """
    base_url = base_url.rstrip("/")

    # Build headers using unified utility
    headers = build_auth_headers(api_key)
    # Remove Content-Type for GET request
    headers.pop("Content-Type", None)

    timeout = aiohttp.ClientTimeout(total=30)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        # Try Ollama /api/tags first
        is_ollama = ":11434" in base_url or "ollama" in base_url.lower()
        if is_ollama:
            try:
                # Construct the tags URL safely, removing a trailing /v1 if present
                base = base_url.rstrip("/")
                if base.endswith("/v1"):
                    base = base[: -len("/v1")]
                ollama_url = f"{base}/api/tags"
                async with session.get(ollama_url, headers=headers) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if "models" in data:
                            return collect_model_names(data["models"])
            except Exception:
                pass  # Fail silently if model fetching fails for this endpoint

        # Try OpenAI-compatible /models
        try:
            models_url = f"{base_url}/models"
            async with session.get(models_url, headers=headers) as resp:
                if resp.status == 200:
                    data = await resp.json()

                    # Handle different response formats
                    if "data" in data and isinstance(data["data"], list):
                        return collect_model_names(data["data"])
                    elif "models" in data and isinstance(data["models"], list):
                        return collect_model_names(data["models"])
                    elif isinstance(data, list):
                        return collect_model_names(data)
        except Exception as e:
            logger.exception("Error fetching models from %s", base_url)

        return []


__all__ = [
    "complete",
    "stream",
    "fetch_models",
]
