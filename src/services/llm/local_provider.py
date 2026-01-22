"""Local LLM provider.

Handles local and self-hosted LLM calls with streaming support.
"""

import asyncio
from collections.abc import AsyncGenerator
import json
from typing import Any

import httpx

from src.logging import get_logger

# Compatibility exports for tests/legacy imports.

logger = get_logger(__name__)

from src.logging import get_logger

from .exceptions import LLMAPIError, LLMConfigError

logger = get_logger(__name__)
from .utils import (
    collect_model_names,
    build_auth_headers,
    build_chat_url,
    clean_thinking_tags,
    extract_response_content,
    sanitize_url,
)

logger = logging.getLogger(__name__)

# Extended timeout for local servers (may be slower than cloud)
DEFAULT_TIMEOUT = 300  # 5 minutes


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
            return content

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
                                        thinking_buffer = content
                                        continue
                                    elif in_thinking_block:
                                        thinking_buffer += content
                                        if "</think>" in thinking_buffer:
                                            # End of thinking block, clean and yield
                                            cleaned = clean_thinking_tags(thinking_buffer)
                                            if cleaned:
                                                yield cleaned
                                            in_thinking_block = False
                                            thinking_buffer = ""
                                        continue
                                    else:
                                        yield content

                        except json.JSONDecodeError:
                            # Non-JSON response, might be raw text
                            if data_str and not data_str.startswith("{"):
                                yield data_str

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
        logger.error(f"⚠️ Streaming failed ({e}), falling back to non-streaming")

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
                resp = await client.get(ollama_url, headers=headers)
                if resp.status_code == 200:
                    data = resp.json()
                    if "models" in data:
                        return [m["name"] for m in data.get("models", [])]
            except Exception as exc:
                logger.debug("Ollama model fetch failed, trying /models: %s", exc)

        # Try OpenAI-compatible /models
        try:
            models_url = f"{base_url}/models"
            resp = await client.get(models_url, headers=headers)
            if resp.status_code == 200:
                data = resp.json()
                # Handle different response formats
                if "data" in data and isinstance(data["data"], list):
                    models: list[str] = []
                    for model_entry in data["data"]:
                        if not isinstance(model_entry, dict):
                            continue
                        model_id = model_entry.get("id") or model_entry.get("name")
                        if model_id:
                            models.append(str(model_id))
                    return models

                if "models" in data and isinstance(data["models"], list):
                    models_list: list[str] = []
                    if data["models"] and isinstance(data["models"][0], dict):
                        for model_entry in data["models"]:
                            if not isinstance(model_entry, dict):
                                continue
                            model_id = model_entry.get("id") or model_entry.get("name")
                            if model_id:
                                models_list.append(str(model_id))
                    else:
                        models_list = [str(model_entry) for model_entry in data["models"]]
                    return models_list

                if isinstance(data, list):
                    models_from_list: list[str] = []
                    for model_entry in data:
                        if isinstance(model_entry, dict):
                            model_id = model_entry.get("id") or model_entry.get("name")
                            if model_id:
                                models_from_list.append(str(model_id))
                        else:
                            models_from_list.append(str(model_entry))
                    return models_from_list
        except Exception as e:
            logger.error(f"Error fetching models from {base_url}: {e}")
            logger.error(f"Error fetching models from {base_url}: {e}")

        return []


__all__ = [
    "complete",
    "stream",
    "fetch_models",
]
