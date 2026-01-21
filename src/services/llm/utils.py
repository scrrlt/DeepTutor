# -*- coding: utf-8 -*-
"""
LLM Utilities
=============

Utility functions for LLM service:
- URL handling for local and cloud servers
- Response content extraction
- Thinking tags cleaning
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


def is_local_llm_server(base_url: str) -> bool:
    """
    Determine whether a URL refers to a local LLM server.
    
    Parameters:
        base_url (str): The base URL to inspect.
    
    Returns:
        bool: `True` if the URL appears to point to a local or private LLM server (local hostname, loopback or private IP range, or common local port), `False` otherwise.
    """
    if not base_url:
        return False

    base_url_lower = base_url.lower()

    # First, exclude known cloud providers
    for domain in CLOUD_DOMAINS:
        if domain in base_url_lower:
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


def _needs_v1_suffix(url: str) -> bool:
    """
    Check if the URL needs /v1 suffix for OpenAI compatibility.

    Most local LLM servers (Ollama, LM Studio, vLLM, llama.cpp) expose
    OpenAI-compatible endpoints at /v1.

    Args:
        url: The URL to check

    Returns:
        True if /v1 should be appended
    """
    if not url:
        return False

    url_lower = url.lower()

    # Skip if already has /v1
    if url_lower.endswith("/v1"):
        return False

    # Only add /v1 for local servers with known ports that need it
    if not is_local_llm_server(url):
        return False

    # Check if URL contains any port that needs /v1 suffix
    # Also check for "ollama" in URL (but not ollama.com cloud service)
    is_ollama = "ollama" in url_lower and "ollama.com" not in url_lower
    if is_ollama:
        return True

    return any(port in url_lower for port in V1_SUFFIX_PORTS)


def sanitize_url(base_url: str, model: str = "") -> str:
    """
    Normalize a base URL for OpenAI-compatible endpoints, ensuring a protocol, removing client-appended endpoint suffixes, and appending `/v1` for qualifying local LLM servers.
    
    Parameters:
        base_url (str): The input base URL; an empty string is returned unchanged.
        model (str): Optional model name (unused; retained for API compatibility).
    
    Returns:
        str: The sanitized base URL suitable for use with OpenAI-compatible clients.
    """
    if not base_url:
        return base_url

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
    Strip reasoning/thinking markers from model-generated text.
    
    Parameters:
        content (str): Raw model output to clean.
        binding (str | None): Optional provider binding name used to decide whether thinking tags apply.
        model (str | None): Optional model name used when determining if thinking tags apply.
    
    Returns:
        str: The input text with reasoning/thinking blocks removed and surrounding whitespace trimmed.
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
    Construct the full chat endpoint URL for the given provider.
    
    Parameters:
        base_url (str): Base endpoint URL (may be a sanitized OpenAI-compatible or Anthropic base).
        api_version (str | None): API version to append as `api-version` query parameter (used by Azure OpenAI).
        binding (str | None): Provider binding name; if "anthropic" or "claude" (case-insensitive), the function uses the `/messages` endpoint, otherwise `/chat/completions` is used.
    
    Returns:
        str: The complete chat endpoint URL with the appropriate path and optional `api-version` query parameter.
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
    Constructs the full completions endpoint URL for an OpenAI-compatible API.
    
    Appends the `/completions` path to the provided base URL (if missing) and, when an `api_version` is supplied, appends it as an `api-version` query parameter. Raises a ValueError for Anthropic/Claude bindings which do not support the legacy completions endpoint.
    
    Parameters:
        base_url (str): Base URL (should be sanitized beforehand; protocol and trailing slash are normalized).
        api_version (str | None): Optional API version to add as `api-version` (commonly used for Azure OpenAI).
        binding (str | None): Optional provider binding name; if `'anthropic'` or `'claude'`, a ValueError is raised.
    
    Returns:
        str: The fully constructed completions endpoint URL.
    
    Raises:
        ValueError: If `binding` is `'anthropic'` or `'claude'`.
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
    
    Accepts a string, a mapping (dict-like) with an "id" or "name" key, or any other object; returns the derived model name string or None when no usable name can be obtained.
    
    Parameters:
        entry: The raw model entry returned by a provider (string, mapping, or other).
    
    Returns:
        The normalized model name, or `None` if a name cannot be derived.
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
    Collect normalized model names from a sequence of provider entries.
    
    Parameters:
        entries (Sequence[object]): Sequence of provider model entries (strings, mappings, or other types) to normalize.
    
    Returns:
        list[str]: List of normalized model name strings, preserving the input order and omitting entries that cannot be normalized.
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
    Create HTTP headers for authenticating requests to different LLM providers.
    
    Parameters:
        api_key (str | None): API key to include in headers; if None or empty, only Content-Type header is returned.
        binding (str | None): Provider binding name (e.g., "anthropic", "azure", "claude") to select provider-specific header keys.
    
    Returns:
        dict[str, str]: A dictionary of HTTP headers suitable for the target provider.
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