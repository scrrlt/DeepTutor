"""LLM utilities.

URL handling, response extraction, and thinking-tag cleanup helpers.
"""

import ipaddress
import os
import re
from typing import Any
from urllib.parse import urlparse

# Known cloud provider domains (should never be treated as local)
CLOUD_DOMAINS = [
    ".openai.com",
    ".anthropic.com",
    ".deepseek.com",
    ".openrouter.ai",
    ".azure.com",
    ".googleapis.com",
    ".cohere.ai",
    ".mistral.ai",
    ".together.ai",
    ".fireworks.ai",
    ".groq.com",
    ".perplexity.ai",
]

# Common local server ports
LOCAL_PORTS = [
    ":1234",  # LM Studio
    ":11434",  # Ollama
    ":8000",  # vLLM
    ":8080",  # llama.cpp
    ":5000",  # Common dev port
    ":3000",  # Common dev port
    ":8001",  # Alternative vLLM
    ":5001",  # Alternative dev port
]

# Local hostname indicators
LOCAL_HOSTS = [
    "localhost",
    "127.0.0.1",
    "0.0.0.0",  # nosec B104
]

# Ports that need /v1 suffix for OpenAI compatibility
V1_SUFFIX_PORTS = {
    ":11434",  # Ollama
    ":1234",  # LM Studio
    ":8000",  # vLLM
    ":8001",  # Alternative vLLM
    ":8080",  # llama.cpp
}


def is_local_llm_server(base_url: str) -> bool:
    """
    Check if the given URL points to a local LLM server.
    Uses robust URL parsing and IP address checking.

    Args:
        base_url: The base URL to check

    Returns:
        True if the URL appears to be a local LLM server
    """
    if not base_url:
        return False

    try:
        parsed = urlparse(base_url)
        hostname = parsed.hostname
        if not hostname:
            return False

        # 1. Check strict string matches
        if hostname in ("localhost", "0.0.0.0", "::1"):
            return True

        # 2. Check IP address ranges (127.0.0.0/8, etc)
        try:
            ip = ipaddress.ip_address(hostname)
            return ip.is_loopback or ip.is_private
        except ValueError:
            # It's a domain name (e.g. google.com)
            pass

        # 3. Fallback to check if it's NOT a known cloud domain
        # (Original logic relied on this, keeping it for hybrid safety)
        base_url_lower = base_url.lower()
        for domain in CLOUD_DOMAINS:
            if domain in base_url_lower:
                return False
        
        # If it's not a known cloud domain, and not an IP... 
        # relying on port might be useful still
        for port in LOCAL_PORTS:
            if port in base_url_lower:
                return True

        return False

    except Exception:
        return False


def sanitize_url(base_url: str, model: str = "") -> str:
    """
    Normalize URL without guessing based on ports.
    
    Args:
        base_url: The base URL to sanitize
        model: Optional model name (unused, kept for API compatibility)

    Returns:
        Sanitized URL string
    """
    if not base_url:
        return ""

    # Force protocol
    if not re.match(r"^[a-zA-Z]+://", base_url):
        base_url = f"http://{base_url}"

    url = base_url.rstrip("/")

    # Strip known endpoints to get back to base
    # e.g. http://localhost:11434/api/chat -> http://localhost:11434/api
    suffixes = [
        "/chat/completions",
        "/messages",
        "/v1",
        "/completions",
        "/embeddings",
    ]
    for suffix in suffixes:
        if url.endswith(suffix):
            url = url[: -len(suffix)]
            url = url.rstrip("/")

    return url


def clean_thinking_tags(
    content: str,
    binding: str | None = None,
    model: str | None = None,
) -> str:
    """
    Remove thinking tags from model output.

    Some reasoning models (DeepSeek, Qwen, etc.) include <think>...</think> blocks
    that should be stripped from the final response.

    Args:
        content: Raw model output
        binding: Provider binding name (optional, for capability check)
        model: Model name (optional, for capability check)

    Returns:
        Cleaned content without thinking tags
    """
    if not content:
        return content

    # Check if model produces thinking tags (if binding/model provided)
    if binding:
        # Lazy import to avoid circular dependency
        from .capabilities import has_thinking_tags

        if not has_thinking_tags(binding, model):
            return content

    # Remove <think>...</think> blocks
    # Note: This regex is simple and doesn't handle streaming well.
    # Future improvements should use a streaming-aware parser.
    if "<think>" in content:
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)

    return content.strip()


def build_chat_url(
    base_url: str,
    api_version: str | None = None,
    binding: str | None = None,
) -> str:
    """
    Build the full chat completions endpoint URL.

    Handles:
    - Adding /chat/completions suffix for OpenAI-compatible endpoints
    - Adding /messages suffix for Anthropic endpoints
    - Adding api-version query parameter for Azure OpenAI

    Args:
        base_url: Base URL (should be sanitized first)
        api_version: API version for Azure OpenAI (optional)
        binding: Provider binding name (optional, for Anthropic detection)

    Returns:
        Full endpoint URL
    """
    if not base_url:
        return base_url

    url = base_url.rstrip("/")

    # Anthropic uses /messages endpoint
    binding_lower = (binding or "").lower()
    if binding_lower in ["anthropic", "claude"]:
        if not url.endswith("/messages"):
            url += "/messages"
    else:
        # OpenAI-compatible endpoints use /chat/completions
        if not url.endswith("/chat/completions"):
            url += "/chat/completions"

    # Add api-version for Azure OpenAI
    if api_version:
        separator = "&" if "?" in url else "?"
        url += f"{separator}api-version={api_version}"

    return url


def extract_response_content(message: dict[str, Any]) -> str:
    """
    Extract content from LLM response message.

    Handles different response formats from various models:
    - Standard content field
    - Reasoning models that use reasoning_content, reasoning, or thought fields

    Args:
        message: Message dict from LLM response (e.g., choices[0].message)

    Returns:
        Extracted content string
    """
    if not message:
        return ""

    # 1. Standard Content
    if content := message.get("content"):
        return content

    # 2. DeepSeek/Reasoning variants (often in 'reasoning_content')
    # If the user *wants* the reasoning, this logic hides it.
    # Ensure this aligns with your design goal (hiding vs showing thought).
    for key in ["reasoning_content", "thought", "reasoning"]:
        if val := message.get(key):
            return val

    # 3. Tool Calls (Don't return empty string if it's a tool call)
    if message.get("tool_calls"):
        return "<tool_call>"

    return ""


def build_auth_headers(
    api_key: str | None,
    binding: str | None = None,
) -> dict[str, str]:
    """
    Build authentication headers for LLM API requests.

    Args:
        api_key: API key
        binding: Provider binding name (for provider-specific headers)

    Returns:
        Headers dict
    """
    headers = {"Content-Type": "application/json"}

    if not api_key:
        return headers

    binding_lower = (binding or "").lower()

    if binding_lower in ["anthropic", "claude"]:
        headers["x-api-key"] = api_key
        # Use explicit version configurable via env if needed, defaulting to known stable
        headers["anthropic-version"] = os.getenv(
            "ANTHROPIC_API_VERSION", "2023-06-01"
        )
    elif binding_lower == "azure_openai":
        headers["api-key"] = api_key
    else:
        headers["Authorization"] = f"Bearer {api_key}"

    return headers


__all__ = [
    # URL utilities
    "sanitize_url",
    "is_local_llm_server",
    "build_chat_url",
    "build_auth_headers",
    # Content utilities
    "clean_thinking_tags",
    "extract_response_content",
    # Constants
    "CLOUD_DOMAINS",
    "LOCAL_PORTS",
    "LOCAL_HOSTS",
    "V1_SUFFIX_PORTS",
]
