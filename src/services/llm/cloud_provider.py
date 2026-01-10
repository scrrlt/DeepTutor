"""
Cloud LLM Provider
==================

Handles all cloud API LLM calls (OpenAI, DeepSeek, Anthropic, etc.)
Provides both complete() and stream() methods.
"""

import logging
import os
from typing import AsyncGenerator, Dict, List, Optional

import aiohttp
from lightrag.llm.openai import openai_complete_if_cache

# Get loggers for suppression during fallback scenarios
# (lightrag logs errors internally before raising exceptions)
_lightrag_logger = logging.getLogger("lightrag")
_openai_logger = logging.getLogger("openai")

from .capabilities import supports_response_format
from .config import get_token_limit_kwargs
from .exceptions import LLMAPIError, LLMAuthenticationError, LLMConfigError
from .utils import (
    build_auth_headers,
    build_chat_url,
    clean_thinking_tags,
    extract_response_content,
    sanitize_url,
)


async def complete(
    prompt: str,
    system_prompt: str = "You are a helpful assistant.",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    api_version: Optional[str] = None,
    binding: str = "openai",
    **kwargs,
) -> str:
    """
    Complete a prompt using cloud API providers.

    Supports OpenAI-compatible APIs, Anthropic, and Gemini.

    Args:
        prompt: The user prompt
        system_prompt: System prompt for context
        model: Model name
        api_key: API key
        base_url: Base URL for the API
        api_version: API version for Azure OpenAI
        binding: Provider binding type (openai, anthropic, gemini)
        **kwargs: Additional parameters (temperature, max_tokens, etc.)

    Returns:
        str: The LLM response
    """
    binding_lower = (binding or "openai").lower()

    if binding_lower in ["anthropic", "claude"]:
        return await _anthropic_complete(
            model=model,
            prompt=prompt,
            system_prompt=system_prompt,
            api_key=api_key,
            base_url=base_url,
            **kwargs,
        )
    
    if binding_lower == "gemini":
        return await _gemini_complete(
            model=model,
            prompt=prompt,
            system_prompt=system_prompt,
            api_key=api_key,
            base_url=base_url,
            messages=kwargs.get("messages"),
            **kwargs,
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
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    api_version: Optional[str] = None,
    binding: str = "openai",
    messages: Optional[List[Dict[str, str]]] = None,
    **kwargs,
) -> AsyncGenerator[str, None]:
    """
    Stream a response from cloud API providers.

    Args:
        prompt: User prompt
        system_prompt: System prompt
        model: Model name
        api_key: API key
        base_url: Base URL
        api_version: API version for Azure OpenAI
        binding: Provider binding type (openai, anthropic, gemini)
        messages: Pre-built messages array (optional)
        **kwargs: Additional parameters

    Yields:
        str: Response chunks
    """
    binding_lower = (binding or "openai").lower()

    if binding_lower in ["anthropic", "claude"]:
        async for chunk in _anthropic_stream(
            model=model,
            prompt=prompt,
            system_prompt=system_prompt,
            api_key=api_key,
            base_url=base_url,
            messages=messages,
            **kwargs,
        ):
            yield chunk
    elif binding_lower == "gemini":
        async for chunk in _gemini_stream(
            model=model,
            prompt=prompt,
            system_prompt=system_prompt,
            api_key=api_key,
            base_url=base_url,
            messages=kwargs.get("messages"),
            **kwargs,
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
    api_key: Optional[str],
    base_url: Optional[str],
    api_version: Optional[str] = None,
    binding: str = "openai",
    **kwargs,
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
        lightrag_kwargs = {
            "system_prompt": system_prompt,
            "history_messages": [],  # Required by lightrag to build messages array
            "api_key": api_key,
            "base_url": base_url,
            **kwargs,
        }
        if api_version:
            lightrag_kwargs["api_version"] = api_version

        # Suppress lightrag's and openai's internal error logging during the call
        # (errors are handled by our fallback mechanism)
        original_lightrag_level = _lightrag_logger.level
        original_openai_level = _openai_logger.level
        _lightrag_logger.setLevel(logging.CRITICAL)
        _openai_logger.setLevel(logging.CRITICAL)
        try:
            # model and prompt must be positional arguments
            content = await openai_complete_if_cache(model, prompt, **lightrag_kwargs)
        finally:
            _lightrag_logger.setLevel(original_lightrag_level)
            _openai_logger.setLevel(original_openai_level)
    except Exception:
        pass  # Fall through to direct call

    # Fallback: Direct aiohttp call
    if not content and base_url:
        # Build URL using unified utility (use binding for Azure detection)
        url = build_chat_url(base_url, api_version, binding)

        # Build headers using unified utility
        headers = build_auth_headers(api_key, binding)

        data = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": kwargs.get("temperature", 0.7),
        }

        # Handle max_tokens / max_completion_tokens based on model
        max_tokens = kwargs.get("max_tokens") or kwargs.get("max_completion_tokens") or 4096
        data.update(get_token_limit_kwargs(model, max_tokens))

        # Include response_format if present in kwargs
        if "response_format" in kwargs:
            data["response_format"] = kwargs["response_format"]

        timeout = aiohttp.ClientTimeout(total=120)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, headers=headers, json=data) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    if "choices" in result and result["choices"]:
                        msg = result["choices"][0].get("message", {})
                        # Use unified response extraction
                        content = extract_response_content(msg)
                else:
                    error_text = await resp.text()
                    raise LLMAPIError(
                        f"OpenAI API error: {error_text}",
                        status_code=resp.status,
                        provider=binding or "openai",
                    )

    if content is not None:
        # Clean thinking tags from response using unified utility
        return clean_thinking_tags(content, binding, model)

    raise LLMConfigError("Cloud completion failed: no valid configuration")


async def _openai_stream(
    model: str,
    prompt: str,
    system_prompt: str,
    api_key: Optional[str],
    base_url: Optional[str],
    api_version: Optional[str] = None,
    binding: str = "openai",
    messages: Optional[List[Dict[str, str]]] = None,
    **kwargs,
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

    data = {
        "model": model,
        "messages": msg_list,
        "temperature": kwargs.get("temperature", 0.7),
        "stream": True,
    }

    # Handle max_tokens / max_completion_tokens based on model
    max_tokens = kwargs.get("max_tokens") or kwargs.get("max_completion_tokens")
    if max_tokens:
        data.update(get_token_limit_kwargs(model, max_tokens))

    # Include response_format if present in kwargs
    if "response_format" in kwargs:
        data["response_format"] = kwargs["response_format"]

    timeout = aiohttp.ClientTimeout(total=300)
    async with aiohttp.ClientSession(timeout=timeout) as session:
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
    api_key: Optional[str],
    base_url: Optional[str],
    messages: Optional[List[Dict[str, str]]] = None,
    **kwargs,
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

    data = {
        "model": model,
        "system": system_content,
        "messages": msg_list,
        "max_tokens": kwargs.get("max_tokens", 4096),
        "temperature": kwargs.get("temperature", 0.7),
    }

    timeout = aiohttp.ClientTimeout(total=120)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(url, headers=headers, json=data) as response:
            if response.status != 200:
                error_text = await response.text()
                raise LLMAPIError(
                    f"Anthropic API error: {error_text}",
                    status_code=response.status,
                    provider="anthropic",
                )

            result = await response.json()
            return result["content"][0]["text"]


async def _anthropic_stream(
    model: str,
    prompt: str,
    system_prompt: str,
    api_key: Optional[str],
    base_url: Optional[str],
    messages: Optional[List[Dict[str, str]]] = None,
    **kwargs,
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

    data = {
        "model": model,
        "system": system_content,
        "messages": msg_list,
        "max_tokens": kwargs.get("max_tokens", 4096),
        "temperature": kwargs.get("temperature", 0.7),
        "stream": True,
    }

    timeout = aiohttp.ClientTimeout(total=300)
    async with aiohttp.ClientSession(timeout=timeout) as session:
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
                    chunk_data = json.loads(data_str)
                    event_type = chunk_data.get("type")
                    if event_type == "content_block_delta":
                        delta = chunk_data.get("delta", {})
                        text = delta.get("text")
                        if text:
                            yield text
                except json.JSONDecodeError:
                    continue


async def _gemini_complete(
    model: str,
    prompt: str,
    system_prompt: str,
    api_key: Optional[str],
    base_url: Optional[str],
    messages: Optional[List[Dict[str, str]]] = None,
    **kwargs,
) -> str:
    """Google Gemini API completion with multimodal support."""
    api_key = api_key or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise LLMAuthenticationError("Gemini API key is missing.", provider="gemini")

    # Build URL - Gemini uses /models/{model}:generateContent
    effective_base = base_url or "https://generativelanguage.googleapis.com/v1beta"
    url = f"{effective_base.rstrip('/')}/models/{model}:generateContent?key={api_key}"

    headers = {"Content-Type": "application/json"}

    # Build parts - handle multimodal content if messages provided
    parts = []
    if messages:
        # Convert from OpenAI format to Gemini format
        last_msg = messages[-1] if messages else None
        if last_msg and isinstance(last_msg.get("content"), list):
            for item in last_msg["content"]:
                if item.get("type") == "text":
                    parts.append({"text": item["text"]})
                elif item.get("type") == "image_url":
                    # Extract base64 data from data URL
                    img_url = item["image_url"]["url"]
                    if "," in img_url:
                        img_data = img_url.split(",", 1)[1]
                        parts.append({
                            "inline_data": {
                                "mime_type": "image/jpeg",
                                "data": img_data
                            }
                        })
    
    if not parts:
        parts = [{"text": prompt}]

    # Build request body - Gemini format
    data = {
        "contents": [{"parts": parts}],
        "safetySettings": [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ],
        "generationConfig": {
            "temperature": kwargs.get("temperature", 0.7),
            "maxOutputTokens": kwargs.get("max_tokens", 4096),
        }
    }

    # Add system instruction if provided
    if system_prompt and system_prompt != "You are a helpful assistant.":
        data["systemInstruction"] = {"parts": [{"text": system_prompt}]}

    timeout = aiohttp.ClientTimeout(total=120)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(url, headers=headers, json=data) as response:
            if response.status != 200:
                error_text = await response.text()
                raise LLMAPIError(
                    f"Gemini API error: {error_text}",
                    status_code=response.status,
                    provider="gemini",
                )

            result = await response.json()
            # Extract text from Gemini response format
            if "candidates" in result and result["candidates"]:
                candidate = result["candidates"][0]
                
                # Check for safety blocking
                finish_reason = candidate.get("finishReason")
                if finish_reason == "SAFETY":
                    raise LLMAPIError(
                        "Gemini blocked response due to safety filters.",
                        provider="gemini",
                    )
                
                if "content" in candidate and "parts" in candidate["content"]:
                    parts = candidate["content"]["parts"]
                    if parts and "text" in parts[0]:
                        return parts[0]["text"]
            
            raise LLMAPIError("Gemini API returned unexpected response format", provider="gemini")


async def _gemini_stream(
    model: str,
    prompt: str,
    system_prompt: str,
    api_key: Optional[str],
    base_url: Optional[str],
    messages: Optional[List[Dict[str, str]]] = None,
    **kwargs,
) -> AsyncGenerator[str, None]:
    """Google Gemini API streaming with multimodal support."""
    import json

    api_key = api_key or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise LLMAuthenticationError("Gemini API key is missing.", provider="gemini")

    # Build URL - Gemini streaming uses :streamGenerateContent
    effective_base = base_url or "https://generativelanguage.googleapis.com/v1beta"
    url = f"{effective_base.rstrip('/')}/models/{model}:streamGenerateContent?key={api_key}&alt=sse"

    headers = {"Content-Type": "application/json"}

    # Build parts - handle multimodal content if messages provided
    parts = []
    if messages:
        last_msg = messages[-1] if messages else None
        if last_msg and isinstance(last_msg.get("content"), list):
            for item in last_msg["content"]:
                if item.get("type") == "text":
                    parts.append({"text": item["text"]})
                elif item.get("type") == "image_url":
                    img_url = item["image_url"]["url"]
                    if "," in img_url:
                        img_data = img_url.split(",", 1)[1]
                        parts.append({
                            "inline_data": {
                                "mime_type": "image/jpeg",
                                "data": img_data
                            }
                        })
    
    if not parts:
        parts = [{"text": prompt}]

    # Build request body
    data = {
        "contents": [{"parts": parts}],
        "safetySettings": [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ],
        "generationConfig": {
            "temperature": kwargs.get("temperature", 0.7),
            "maxOutputTokens": kwargs.get("max_tokens", 4096),
        }
    }

    if system_prompt and system_prompt != "You are a helpful assistant.":
        data["systemInstruction"] = {"parts": [{"text": system_prompt}]}

    timeout = aiohttp.ClientTimeout(total=300)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(url, headers=headers, json=data) as response:
            if response.status != 200:
                error_text = await response.text()
                raise LLMAPIError(
                    f"Gemini stream error: {error_text}",
                    status_code=response.status,
                    provider="gemini",
                )

            async for line in response.content:
                line_str = line.decode("utf-8").strip()
                if not line_str or not line_str.startswith("data:"):
                    continue

                data_str = line_str[5:].strip()
                if not data_str:
                    continue

                try:
                    chunk_data = json.loads(data_str)
                    if "candidates" in chunk_data and chunk_data["candidates"]:
                        candidate = chunk_data["candidates"][0]
                        if "content" in candidate and "parts" in candidate["content"]:
                            parts = candidate["content"]["parts"]
                            if parts and "text" in parts[0]:
                                yield parts[0]["text"]
                except json.JSONDecodeError:
                    continue


async def fetch_models(
    base_url: str,
    api_key: Optional[str] = None,
    binding: str = "openai",
) -> List[str]:
    """
    Fetch available models from cloud provider.

    Args:
        base_url: API endpoint URL
        api_key: API key
        binding: Provider type (openai, anthropic)

    Returns:
        List of available model names
    """
    binding = (binding or "openai").lower()
    base_url = base_url.rstrip("/")

    # Provider-specific shortcuts and request shapes
    if binding in ["anthropic", "claude"]:
        # Anthropic does not expose a list endpoint; return common models
        return [
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022",
            "claude-3-opus-20240229",
        ]

    timeout = aiohttp.ClientTimeout(total=30)

    # Build URL and headers per provider
    if binding == "gemini":
        if not api_key:
            return []
        url = f"{base_url}/models?key={api_key}"
        headers = {"Content-Type": "application/json"}
    elif binding == "azure_openai":
        if not api_key:
            return []
        # Azure lists deployments, not models
        api_version = "2023-05-15"
        url = f"{base_url}/openai/deployments?api-version={api_version}"
        headers = {"api-key": api_key}
    else:
        # OpenAI-compatible
        url = f"{base_url}/models"
        headers = build_auth_headers(api_key, binding)
        headers.pop("Content-Type", None)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        try:
            async with session.get(url, headers=headers) as resp:
                if resp.status != 200:
                    return []

                data = await resp.json()

                # Gemini format: {"models": [{"name": "models/gemini-1.5-flash", ...}]}
                if "models" in data and isinstance(data["models"], list):
                    return [
                        m["name"].replace("models/", "")
                        for m in data["models"]
                        if "name" in m
                    ]

                # Azure deployments format: {"value": [{ "model": "...", "id": "...", "name": "..." }]}
                if "value" in data and isinstance(data["value"], list):
                    return [
                        m.get("model") or m.get("id") or m.get("name")
                        for m in data["value"]
                        if m.get("model") or m.get("id") or m.get("name")
                    ]

                # OpenAI format: {"data": [{"id": "gpt-4", ...}]}
                if "data" in data and isinstance(data["data"], list):
                    return [
                        m.get("id") or m.get("name")
                        for m in data["data"]
                        if m.get("id") or m.get("name")
                    ]

                # Plain list format
                if isinstance(data, list):
                    return [
                        m.get("id") or m.get("name") if isinstance(m, dict) else str(m)
                        for m in data
                    ]

            return []
        except Exception as e:
            print(f"Error fetching models from {base_url}: {e}")
            return []


__all__ = [
    "complete",
    "stream",
    "fetch_models",
]
