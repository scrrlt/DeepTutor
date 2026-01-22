# -*- coding: utf-8 -*-
"""
LLM Utilities
=============

URL handling, response extraction, and thinking-tag cleanup helpers.
"""

from collections.abc import Mapping, Sequence
import re
from typing import Any, cast

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
    "0.0.0.0",  # Used by some local LLM servers for all-interface binding  # nosec B104
]

# Ports that need /v1 suffix for OpenAI compatibility
V1_SUFFIX_PORTS = {
    ":11434",  # Ollama
    ":1234",  # LM Studio
    ":8000",  # vLLM
    ":8001",  # Alternative vLLM
    ":8080",  # llama.cpp
}


def is_local_llm_server(
    base_url: str, allow_private: bool | None = None
) -> bool:
    """
    Check if the given URL points to a local LLM server.

    Detects local servers by:
    1. Checking for local/private hostnames and IPs
    2. Checking for private IP ranges (10.x.x.x, 192.168.x.x, 172.16-31.x.x)
    3. Checking for common local LLM server ports (as fallback)
    4. Excluding known cloud provider domains

    Args:
        base_url: The base URL to check
        allow_private: Optional override to treat private network IPs as local.
                       If None, reads the LLM_TREAT_PRIVATE_AS_LOCAL env var
                       ("1", "true", "yes" => True).

    Returns:
        True if the URL appears to be a local LLM server
    """
    if not base_url:
        return False

    # Resolve allow_private from env if not explicitly provided
    if allow_private is None:
        val = os.environ.get("LLM_TREAT_PRIVATE_AS_LOCAL")
        if val is not None:
            allow_private = val.strip().lower() in ("1", "true", "yes")

    try:
        parsed = urlparse(base_url)
        hostname = parsed.hostname
        if not hostname:
            return False

    # Extract hostname/IP from URL
    try:
        from urllib.parse import urlparse
        parsed = urlparse(base_url)
        hostname = parsed.hostname or parsed.netloc
        if not hostname:
            return False
        hostname_lower = hostname.lower()
    except Exception:
        # Fallback to simple string checks if URL parsing fails
        hostname_lower = base_url_lower

    # Check for local hostname indicators (regardless of port)
    if any(host in hostname_lower for host in LOCAL_HOSTS):
        return True

    # Check for private IP ranges
    import ipaddress
    try:
        # Try to parse as IP address
        ip = ipaddress.ip_address(hostname)
        # Check if it's a private IP
        if ip.is_private:
            return True
        # Also allow loopback IPs
        if ip.is_loopback:
            return True
    except ValueError:
        # Not a valid IP, continue with hostname checks
        pass

    # Check for common local server ports (as fallback for edge cases)
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

    # Ensure URL has a protocol (default to http for local servers)
    if url and not url.startswith(("http://", "https://")):
        url = "http://" + url

    # Standard OpenAI client library is strict about URLs:
    # - No trailing slashes
    # - No /chat/completions or /completions/messages/embeddings suffixes
    #   (it adds these automatically)
    for suffix in [
        "/chat/completions",
        "/completions",
        "/messages",
        "/embeddings",
    ]:
        if url.endswith(suffix):
            url = url[: -len(suffix)]
            url = url.rstrip("/")

    # For local LLM servers, ensure /v1 is present for OpenAI compatibility
    if _needs_v1_suffix(url):
        url = url.rstrip("/") + "/v1"

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
        return ""

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

    # Remove unicode thinking delimiters often used by local/reasoning models
    # e.g., ◣ ... ◢  (U+25E3 / U+25E2) or repeated marker like 꽁...꽁
    # Use non-greedy matching and DOTALL to span multiple lines
    if "◣" in content and "◢" in content:
        content = re.sub(r"\u25E3.*?\u25E2", "", content, flags=re.DOTALL)

    if "꽁" in content:
        # 꽁 is sometimes used as both open+close marker, remove paired blocks
        content = re.sub(r"꽁.*?꽁", "", content, flags=re.DOTALL)

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

    Raises:
        ValueError: If an unsupported binding is provided.
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


def build_completion_url(
    base_url: str,
    api_version: str | None = None,
    binding: str | None = None,
) -> str:
    """
    Build the full completions endpoint URL.

    Handles:
    - Adding /completions suffix for OpenAI-compatible endpoints
    - Adding api-version query parameter for Azure OpenAI

    Args:
        base_url: Base URL (should be sanitized first)
        api_version: API version for Azure OpenAI (optional)
        binding: Provider binding name (optional, for compatibility)

    Returns:
        Full endpoint URL

    Raises:
        ValueError: If binding is 'anthropic' or 'claude' (Anthropic does not
            support the legacy completions endpoint).
    """
    if not base_url:
        return base_url

    url = base_url.rstrip("/")

    binding_lower = (binding or "").lower()
    if binding_lower in ["anthropic", "claude"]:
        raise ValueError("Anthropic does not support /completions endpoint")

    if not url.endswith("/completions"):
        url += "/completions"

    if api_version:
        separator = "&" if "?" in url else "?"
        url += f"{separator}api-version={api_version}"

    return url


def extract_response_content(message: Any) -> str:
    """
    Extract content from LLM response message.

    Handles different response formats from various models:
    - Standard content field
    - Reasoning models that use reasoning_content, reasoning, or thought fields
    - Direct strings or None values

    Args:
        message: Message object/dict from LLM response or direct string

    Returns:
        Extracted content string
    """
    if message is None:
        return ""

    if isinstance(message, str):
        return message

    if not isinstance(message, (dict, Mapping)):
        return str(message)

    content = message.get("content", "")

    # Handle reasoning models that return content in different fields
    if not content:
        content = (
            message.get("reasoning_content")
            or message.get("reasoning")
            or message.get("thought")
            or ""
        )

    return str(content)


def _normalize_model_name(entry: object) -> str | None:
    """
    Normalize a model name from a provider payload entry.

    Args:
        entry: The raw model entry returned by a provider.

    Returns:
        The normalized model name, or None if one cannot be derived.

    Raises:
        None.
    """
    if entry is None:
        return None

    if isinstance(entry, str):
        return entry if entry else None

    if isinstance(entry, Mapping):
        # Use cast to ensure type safety - Mapping interface doesn't guarantee get() method
        # but we know this will be a dict-like object in practice from model APIs
        entry_dict = cast(dict[str, Any], entry)
        name = entry_dict.get("id")
        if name is None:
            name = entry_dict.get("name")
        if name is None:
            return None
        text = str(name)
        return text if text else None

    text = str(entry)
    return text if text else None


def collect_model_names(entries: Sequence[object]) -> list[str]:
    """
    Collect normalized model names from a sequence of entries.

    Args:
        entries: Sequence of model entries from provider payloads.

    Returns:
        List of normalized model names.

    Raises:
        None.
    """
    names: list[str] = []
    for entry in entries:
        name = _normalize_model_name(entry)
        if name is not None:
            names.append(name)
    return names


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

    Raises:
        None.
    """
    headers = {"Content-Type": "application/json"}

    if not api_key:
        return headers

    binding_lower = (binding or "").lower()

    if binding_lower in ["anthropic", "claude"]:
        headers["x-api-key"] = api_key
        headers["anthropic-version"] = "2023-06-01"
    elif binding_lower in ["azure_openai", "azure"]:
        headers["api-key"] = api_key
    else:
        headers["Authorization"] = f"Bearer {api_key}"

    return headers


__all__ = [
    # URL utilities
    "sanitize_url",
    "is_local_llm_server",
    "build_chat_url",
    "build_completion_url",
    "build_auth_headers",
    "collect_model_names",
    # Content utilities
    "clean_thinking_tags",
    "extract_response_content",
    # Constants
    "CLOUD_DOMAINS",
    "LOCAL_PORTS",
    "LOCAL_HOSTS",
    "V1_SUFFIX_PORTS",
]
