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
from dataclasses import dataclass
from functools import lru_cache
import ipaddress
import os
from typing import Any, cast
from urllib.parse import urlparse

DEFAULT_LOCAL_CIDRS = (
    "127.0.0.0/8",
    "10.0.0.0/8",
    "172.16.0.0/12",
    "192.168.0.0/16",
    "::1/128",
)

DEFAULT_LOCAL_HOSTS = {
    "localhost",
    "127.0.0.1",
    "0.0.0.0",  # Used by some local LLM servers for all-interface binding  # nosec B104
    "::1",
}

Network = ipaddress.IPv4Network | ipaddress.IPv6Network


@dataclass(frozen=True, slots=True)
class NetworkSettings:
    """Network detection configuration for local LLM servers.

    Args:
        cloud_domains: Domain suffixes to treat as non-local.
        local_cidrs: CIDR blocks considered local/private.
        local_hosts: Hostnames treated as local.
    """

    cloud_domains: set[str]
    local_cidrs: tuple[Network, ...]
    local_hosts: set[str]


def _parse_csv_env(name: str) -> set[str]:
    raw = os.getenv(name, "")
    return {item.strip().lower() for item in raw.split(",") if item.strip()}


def _parse_cidrs_env(name: str, defaults: Sequence[str]) -> tuple[Network, ...]:
    raw = os.getenv(name, "")
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        values = list(defaults)
    return tuple(ipaddress.ip_network(value) for value in values)


@lru_cache(maxsize=1)
def get_network_settings() -> NetworkSettings:
    """Load network detection settings from environment overrides.

    Returns:
        NetworkSettings with defaults and environment overrides applied.

    Raises:
        ValueError: If configured CIDR values are invalid.
    """
    cloud_domains = _parse_csv_env("LLM_CLOUD_DOMAINS")
    local_hosts = _parse_csv_env("LLM_LOCAL_HOSTS") or set(DEFAULT_LOCAL_HOSTS)
    local_cidrs = _parse_cidrs_env("LLM_LOCAL_CIDRS", DEFAULT_LOCAL_CIDRS)
    return NetworkSettings(
        cloud_domains=cloud_domains,
        local_cidrs=local_cidrs,
        local_hosts=local_hosts,
    )


def is_local_llm_server(base_url: str) -> bool:
    """
    Check if the given URL points to a local LLM server.

    Detects local servers by:
    1. Checking for configured local hostnames
    2. Checking configured local CIDR ranges
    3. Excluding configured cloud provider domains

    Args:
        base_url: The base URL to check

    Returns:
        True if the URL appears to be a local LLM server
    """
    if not base_url:
        return False

    base_url_lower = base_url.lower()
    settings = get_network_settings()

    for domain in settings.cloud_domains:
        if domain and domain in base_url_lower:
            return False

    try:
        parsed = urlparse(base_url)
        hostname = parsed.hostname or parsed.netloc
        if not hostname:
            return False
        hostname_lower = hostname.lower()
    except Exception:
        hostname_lower = base_url_lower

    if any(host in hostname_lower for host in settings.local_hosts):
        return True

    try:
        ip = ipaddress.ip_address(hostname)
        for network in settings.local_cidrs:
            if ip in network:
                return True
    except ValueError:
        pass

    return False


def _needs_v1_suffix(url: str) -> bool:
    """
    Check if the URL needs /v1 suffix for OpenAI compatibility.

    For robustness we treat any detected local LLM server as requiring /v1.
    This avoids depending on a static list of ports which may not match container
    port mappings used in Docker/Kubernetes setups.
    """
    if not url:
        return False

    url_lower = url.lower()

    # Skip if already has /v1
    if url_lower.endswith("/v1"):
        return False

    # If it's an Ollama host explicitly (not the cloud service), ensure /v1
    is_ollama = "ollama" in url_lower and "ollama.com" not in url_lower
    if is_ollama:
        return True

    # For other local servers, rely on the local detection which checks hostnames
    # and private IP ranges instead of a static list of ports.
    return is_local_llm_server(url)


def sanitize_url(base_url: str, model: str = "") -> str:
    """
    Sanitize base URL for OpenAI-compatible APIs, with special handling for local LLM servers.

    Handles:
    - Ollama (port 11434)
    - LM Studio (port 1234)
    - vLLM (port 8000)
    - llama.cpp (port 8080)
    - Other localhost OpenAI-compatible servers

    Args:
        base_url: The base URL to sanitize
        model: Optional model name (unused, kept for API compatibility)

    Returns:
        Sanitized URL string
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

    content = _strip_tagged_content(content, "<think>", "</think>")
    content = _strip_tagged_content(content, "◣", "◢")
    content = _strip_tagged_content(content, "꽁", "꽁")
    return content.strip()


def _strip_tagged_content(content: str, open_tag: str, close_tag: str) -> str:
    """Remove tagged sections without regex backtracking overhead.

    Args:
        content: Input text.
        open_tag: Opening marker.
        close_tag: Closing marker.

    Returns:
        Text with tagged sections removed.

    Raises:
        None.
    """
    if open_tag not in content:
        return content

    segments: list[str] = []
    index = 0
    while True:
        start = content.find(open_tag, index)
        if start == -1:
            segments.append(content[index:])
            break
        segments.append(content[index:start])
        end = content.find(close_tag, start + len(open_tag))
        if end == -1:
            break
        index = end + len(close_tag)

    return "".join(segments)


def _ensure_azure_deployment_path(base_url: str, deployment: str | None) -> str:
    """
    Ensure Azure OpenAI deployment path is present in the base URL.

    Args:
        base_url: Azure endpoint base URL.
        deployment: Azure deployment name (model identifier).

    Returns:
        Base URL with the deployment path appended.

    Raises:
        ValueError: If deployment is missing for Azure bindings.
    """
    if not base_url:
        return base_url

    if not deployment:
        raise ValueError("Azure OpenAI requires a deployment name for routing")

    from urllib.parse import urlparse, urlunparse

    parsed = urlparse(base_url)
    path = parsed.path.rstrip("/")
    if "/openai/deployments/" not in path:
        path = f"{path}/openai/deployments/{deployment}"

    return urlunparse(
        (
            parsed.scheme,
            parsed.netloc,
            path,
            parsed.params,
            parsed.query,
            parsed.fragment,
        )
    )


def build_chat_url(
    base_url: str,
    api_version: str | None = None,
    binding: str | None = None,
    model: str | None = None,
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
        model: Model or deployment name (required for Azure OpenAI)

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
    if binding_lower in ["azure", "azure_openai"]:
        url = _ensure_azure_deployment_path(url, model)
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
    model: str | None = None,
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
        model: Model or deployment name (required for Azure OpenAI)

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

    if binding_lower in ["azure", "azure_openai"]:
        url = _ensure_azure_deployment_path(url, model)

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
        Extracted content string. For unrecognized types returns an empty
        string to avoid poisoning downstream logic with reprs of error objects.
    """
    if message is None:
        return ""

    if isinstance(message, str):
        return message

    # Defensive: non-mapping types (e.g., HTTP response objects) should not be
    # returned as string reprs because that can be mistaken for valid content.
    if not isinstance(message, (dict, Mapping)):
        return ""

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
    # Network configuration
    "get_network_settings",
    "NetworkSettings",
]
