"""Local LLM provider.

Handles local and self-hosted LLM calls with streaming support.
"""

import asyncio
from collections.abc import AsyncGenerator
import json
import logging
from typing import Any, AsyncGenerator, Dict, List, Optional

import aiohttp

from .exceptions import LLMAPIError, LLMConfigError

logger = get_logger(__name__)
from .utils import (
    collect_model_names,
    build_auth_headers,
    build_chat_url,
    clean_thinking_tags,
    collect_model_names,
    extract_response_content,
    sanitize_url,
)

logger = logging.getLogger(__name__)

# Extended timeout for local servers (may be slower than cloud)
DEFAULT_TIMEOUT = 300  # 5 minutes


async def complete(
    prompt: str,
    system_prompt: str = "You are a helpful assistant.",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    messages: Optional[List[Dict[str, str]]] = None,
    **kwargs: Any,
) -> str:
    """
    Complete a prompt using local LLM server.

    Uses httpx async client for local servers.

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
        raise LLMConfigError("base_url is required for local LLM provider")

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

    timeout = httpx.Timeout(kwargs.get("timeout", DEFAULT_TIMEOUT))

    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(url, json=data, headers=headers)
        if response.status_code != 200:
            raise LLMAPIError(
                f"Local LLM error: {response.text}",
                status_code=response.status_code,
                provider="local",
            )

        result = response.json()

            if "choices" in result and result["choices"]:
                msg = result["choices"][0].get("message", {})
                # Use unified response extraction
                content = extract_response_content(msg)
                # Clean thinking tags using unified utility
                content = clean_thinking_tags(content)
                if content:
                    return content

            logger.warning("Local LLM returned no choices: %s", result)
            return ""


async def stream(
    prompt: str,
    system_prompt: str = "You are a helpful assistant.",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    messages: Optional[List[Dict[str, str]]] = None,
    **kwargs: Any,
) -> AsyncGenerator[str, None]:
    """
    Stream a response from local LLM server.

    Uses httpx async client for local servers.
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
        raise LLMConfigError("base_url is required for local LLM provider")

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

    timeout = httpx.Timeout(kwargs.get("timeout", DEFAULT_TIMEOUT))

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream(
                "POST",
                url,
                json=data,
                headers=headers,
            ) as response:
                if response.status_code != 200:
                    error_body = await response.aread()
                    raise LLMAPIError(
                        f"Local LLM error: {error_body.decode('utf-8', errors='replace')}",
                        status_code=response.status_code,
                        provider="local",
                    )

                # Track if we're inside a thinking block
                in_thinking_block = False
                thinking_buffer = ""

                async for line_str in response.aiter_lines():
                    line_str = line_str.strip()

                    # Skip empty lines
                    if not line_str:
                        continue

                    # Handle SSE format
                    if line_str.startswith("data:"):
                        data_str = line_str[5:].strip()

                        if data_str == "[DONE]":
                            break

                        try:
                            chunk_data = json.loads(data_str)
                            if "choices" in chunk_data and chunk_data["choices"]:
                                delta = chunk_data["choices"][0].get("delta", {})
                                content = delta.get("content")

                                if content:
                                    # Handle thinking tags in streaming
                                    if "<think>" in content:
                                        in_thinking_block = True
                                        # Handle case where content has text BEFORE <think>
                                        parts = content.split("<think>", 1)
                                        if parts[0]:
                                            yield parts[0]
                                        thinking_buffer = "<think>" + parts[1]

                                        # Check if closed immediately in same chunk
                                        if "</think>" in thinking_buffer:
                                            cleaned = clean_thinking_tags(thinking_buffer)
                                            if cleaned:
                                                yield cleaned
                                            thinking_buffer = ""
                                            in_thinking_block = False
                                        continue
                                    elif in_thinking_block:
                                        thinking_buffer += content
                                        if "</think>" in thinking_buffer:
                                            # Block finished
                                            cleaned = clean_thinking_tags(thinking_buffer)
                                            if cleaned:
                                                yield cleaned
                                            in_thinking_block = False
                                            thinking_buffer = ""
                                        continue
                                    else:
                                        yield content

                        except json.JSONDecodeError:
                            # Log and skip malformed JSON chunks
                            logger.warning(f"Skipping malformed JSON chunk: {data_str[:50]}...")
                            continue

                    # Some servers don't use SSE format
                    elif line_str.startswith("{"):
                        try:
                            chunk_data = json.loads(line_str)
                            if "choices" in chunk_data and chunk_data["choices"]:
                                delta = chunk_data["choices"][0].get("delta", {})
                                content = delta.get("content")
                                if content:
                                    # TODO: Implement <think> tag parsing for non-SSE JSON streams if supported
                                    yield content
                        except json.JSONDecodeError:
                            pass

    except LLMAPIError:
        raise  # Re-raise LLM errors as-is
    except asyncio.CancelledError:
        raise
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
        except asyncio.CancelledError:
            raise
        except Exception as e2:
            raise LLMAPIError(
                f"Local LLM failed: streaming={e}, non-streaming={e2}",
                provider="local",
            )


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

    timeout = httpx.Timeout(30.0)

    async with httpx.AsyncClient(timeout=timeout) as client:
        # Try Ollama /api/tags first
        is_ollama = ":11434" in base_url or "ollama" in base_url.lower()
        if is_ollama:
            try:
                ollama_url = base_url.replace("/v1", "") + "/api/tags"
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
            logger.error("Error fetching models from %s: %s", base_url, e)

        return []


__all__ = [
    "complete",
    "stream",
    "fetch_models",
]
