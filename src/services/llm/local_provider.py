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

import json
import logging
import os
from typing import Any, AsyncGenerator, Dict, List, Optional

import aiohttp

from .chat_templates import ChatTemplateInfo, render_chat_template, resolve_chat_template
from .exceptions import LLMAPIError, LLMConfigError
from .utils import (
    build_auth_headers,
    build_chat_url,
    build_completion_url,
    clean_thinking_tags,
    collect_model_names,
    extract_response_content,
    sanitize_url,
)

logger = logging.getLogger(__name__)

# Extended timeout for local servers (may be slower than cloud)
DEFAULT_TIMEOUT = 300  # 5 minutes
TEMPLATE_MODE_ENV = "LOCAL_LLM_TEMPLATE_MODE"
TEMPLATE_MODE_COMPLETIONS = "completions"
TEMPLATE_MODE_CHAT = "chat"


def _process_thinking_content(
    content: str,
    in_thinking_block: bool,
    thinking_buffer: str,
) -> tuple[bool, str, str | None]:
    """
    Strip <think> blocks from streaming content while preserving user output.

    Args:
        content: Incoming content chunk.
        in_thinking_block: Whether we're currently inside a <think> block.
        thinking_buffer: Buffer for accumulating <think> content.

    Returns:
        Tuple of updated (in_thinking_block, thinking_buffer, output_chunk).

    Raises:
        None.
    """
    if "<think>" in content:
        idx = content.index("<think>")
        prefix = content[:idx]
        in_thinking_block = True
        thinking_buffer += content[idx:]
        if prefix:
            return in_thinking_block, thinking_buffer, prefix
    elif in_thinking_block:
        thinking_buffer += content
    else:
        return in_thinking_block, thinking_buffer, content

    if "</think>" in thinking_buffer:
        cleaned = clean_thinking_tags(thinking_buffer)
        in_thinking_block = False
        thinking_buffer = ""
        return in_thinking_block, thinking_buffer, cleaned or None

    return in_thinking_block, thinking_buffer, None


def _process_stream_chunk(
    chunk_data: dict[str, Any],
    template_mode: str | None,
    apply_template: bool,
    in_thinking_block: bool,
    thinking_buffer: str,
) -> tuple[bool, str, str | None]:
    """
    Process a stream chunk and return updated state plus output.

    Args:
        chunk_data: Parsed JSON chunk data.
        template_mode: Template mode override when templates are used.
        apply_template: Whether a template is applied for completions.
        in_thinking_block: Whether the stream is inside a <think> block.
        thinking_buffer: Accumulated <think> content buffer.

    Returns:
        Updated (in_thinking_block, thinking_buffer, output_chunk).

    Raises:
        None.
    """
    mode = template_mode if apply_template and template_mode else TEMPLATE_MODE_CHAT
    content = _extract_stream_delta(chunk_data, mode)
    if content:
        return _process_thinking_content(content, in_thinking_block, thinking_buffer)
    return in_thinking_block, thinking_buffer, None


def _resolve_template_info(
    model: str | None,
    kwargs: dict[str, Any],
) -> tuple[ChatTemplateInfo | None, str | None, bool]:
    """
    Resolve chat template configuration and mode.

    Args:
        model: Model name.
        kwargs: Mutable keyword arguments from the caller.

    Returns:
        Tuple of (template_info, template_mode, apply_template).

    Raises:
        None.
    """
    template = kwargs.pop("chat_template", None)
    template_path = kwargs.pop("chat_template_path", None)
    tokenizer_dir = kwargs.pop("tokenizer_dir", None)
    model_dir = kwargs.pop("model_dir", None)
    template_dir = kwargs.pop("template_dir", None)
    apply_template = kwargs.pop("apply_chat_template", None)

    if apply_template is None:
        apply_template = any(
            (
                template,
                template_path,
                tokenizer_dir,
                model_dir,
                os.getenv("LOCAL_LLM_CHAT_TEMPLATE"),
                os.getenv("LOCAL_LLM_CHAT_TEMPLATE_PATH"),
                os.getenv("LOCAL_LLM_TOKENIZER_DIR"),
                os.getenv("LOCAL_LLM_MODEL_DIR"),
            )
        )

    if not apply_template:
        return None, None, False

    template_info = resolve_chat_template(
        model=model,
        template=template,
        template_path=template_path,
        tokenizer_dir=tokenizer_dir,
        model_dir=model_dir,
        template_dir=template_dir,
    )

    if not template_info:
        return None, None, False

    template_mode = kwargs.pop("chat_template_mode", None) or os.getenv(
        TEMPLATE_MODE_ENV, TEMPLATE_MODE_COMPLETIONS
    )
    template_mode = template_mode.lower()

    return template_info, template_mode, True


def _extract_completion_text(payload: dict[str, Any]) -> str:
    """
    Extract text from a completion response payload.

    Args:
        payload: Raw response payload.

    Returns:
        Extracted completion text.

    Raises:
        None.
    """
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""

    first_choice = choices[0]
    if isinstance(first_choice, dict):
        text = first_choice.get("text")
        if isinstance(text, str):
            return text
        message = first_choice.get("message", {})
        return extract_response_content(message)

    return ""


def _extract_stream_delta(
    payload: dict[str, Any],
    template_mode: str,
) -> str | None:
    """
    Extract streaming delta content based on the API mode.

    Args:
        payload: Parsed SSE chunk payload.
        template_mode: Template mode (chat or completions).

    Returns:
        Extracted delta text, if any.

    Raises:
        None.
    """
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return None

    choice = choices[0]
    if not isinstance(choice, dict):
        return None

    if template_mode == TEMPLATE_MODE_COMPLETIONS:
        text = choice.get("text")
        if isinstance(text, str) and text:
            return text
        delta = choice.get("delta", {})
        if isinstance(delta, dict):
            text = delta.get("text")
            if isinstance(text, str) and text:
                return text
        return None

    delta = choice.get("delta", {})
    if isinstance(delta, dict):
        text = delta.get("content")
        if isinstance(text, str) and text:
            return text

    return None


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
        raise LLMConfigError("base_url is required for local LLM provider")

    # Sanitize URL and build endpoint
    base_url = sanitize_url(base_url, model or "")

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

    template_info, template_mode, apply_template = _resolve_template_info(
        model,
        kwargs,
    )

    if apply_template and template_info and template_mode:
        rendered_prompt = render_chat_template(
            msg_list,
            template_info,
            add_generation_prompt=kwargs.pop(
                "add_generation_prompt",
                True,
            ),
        )
        url = build_completion_url(base_url)
        data = {
            "model": model or "default",
            "prompt": rendered_prompt,
            "temperature": kwargs.get("temperature", 0.7),
            "stream": False,
        }
    else:
        url = build_chat_url(base_url)
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

    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(url, json=data, headers=headers) as response:
            if response.status != 200:
                error_text = await response.text()
                raise LLMAPIError(
                    f"Local LLM error: {error_text}",
                    status_code=response.status,
                    provider="local",
                )

            result = await response.json()

            if "choices" in result and result["choices"]:
                if apply_template and template_mode == TEMPLATE_MODE_COMPLETIONS:
                    content = _extract_completion_text(result)
                else:
                    msg = result["choices"][0].get("message", {})
                    content = extract_response_content(msg)
                content = clean_thinking_tags(content)
                return content

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
        raise LLMConfigError("base_url is required for local LLM provider")

    # Sanitize URL and build endpoint
    base_url = sanitize_url(base_url, model or "")

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

    template_info, template_mode, apply_template = _resolve_template_info(
        model,
        kwargs,
    )

    if apply_template and template_info and template_mode:
        rendered_prompt = render_chat_template(
            msg_list,
            template_info,
            add_generation_prompt=kwargs.pop(
                "add_generation_prompt",
                True,
            ),
        )
        url = build_completion_url(base_url)
        data = {
            "model": model or "default",
            "prompt": rendered_prompt,
            "temperature": kwargs.get("temperature", 0.7),
            "stream": True,
        }
    else:
        url = build_chat_url(base_url)
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
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, json=data, headers=headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise LLMAPIError(
                        f"Local LLM stream error: {error_text}",
                        status_code=response.status,
                        provider="local",
                    )

                # Track if we're inside a thinking block
                in_thinking_block = False
                thinking_buffer = ""

                async for line in response.content:
                    line_str = line.decode("utf-8").strip()

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
                            (
                                in_thinking_block,
                                thinking_buffer,
                                output_chunk,
                            ) = _process_stream_chunk(
                                chunk_data,
                                template_mode,
                                apply_template,
                                in_thinking_block,
                                thinking_buffer,
                            )
                            if output_chunk:
                                yield output_chunk

                        except json.JSONDecodeError:
                            # Non-JSON response, might be raw text
                            if data_str and not data_str.startswith("{"):
                                yield data_str

                    # Some servers don't use SSE format
                    elif line_str.startswith("{"):
                        try:
                            chunk_data = json.loads(line_str)
                            (
                                in_thinking_block,
                                thinking_buffer,
                                output_chunk,
                            ) = _process_stream_chunk(
                                chunk_data,
                                template_mode,
                                apply_template,
                                in_thinking_block,
                                thinking_buffer,
                            )
                            if output_chunk:
                                yield output_chunk
                        except json.JSONDecodeError:
                            pass
                    # Handle any remaining raw text chunks from non-SSE/non-JSON responses
                    elif line_str:
                        # Always clean raw text chunks to strip any thinking tags if present
                        cleaned_text = clean_thinking_tags(line_str)
                        if cleaned_text:
                            yield cleaned_text

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
            raise LLMAPIError(
                f"Local LLM failed: streaming={e}, non-streaming={e2}",
                provider="local",
            )


async def fetch_models(
    base_url: str,
    api_key: Optional[str] = None,
) -> List[str]:
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
                ollama_url = base_url.replace("/v1", "") + "/api/tags"
                async with session.get(ollama_url, headers=headers) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if "models" in data:
                            return collect_model_names(data["models"])
            except Exception:
                # Fail silently if model fetching fails for this endpoint
                # We continue trying other endpoints or return empty list
                pass  # nosec B110

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
