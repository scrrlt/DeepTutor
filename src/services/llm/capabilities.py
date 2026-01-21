# -*- coding: utf-8 -*-
"""
Provider Capabilities
=====================

Centralized configuration for LLM provider capabilities.
Optimized for performance via pre-sorted lookups and enforced immutability.
Refactored for thread safety and type integrity.
"""

from __future__ import annotations

import functools
import threading
import warnings
from types import MappingProxyType
from typing import Any, Mapping, Optional, Tuple

# --- Constants & Defaults ---
DEFAULT_PROVIDER = "openai"
DEFAULT_OPEN_MARKERS: Tuple[str, ...] = ("<think>",)
DEFAULT_CLOSE_MARKERS: Tuple[str, ...] = ("</think>",)

# --- Configuration ---

# Frozen provider capabilities.
# Type hint uses MappingProxyType to reflect the actual immutable implementation.
PROVIDER_CAPABILITIES: MappingProxyType[str, MappingProxyType[str, Any]] = MappingProxyType(
    {
        "openai": MappingProxyType(
            {
                "supports_response_format": True,
                "supports_streaming": True,
                "supports_tools": True,
                "system_in_messages": True,
                "newer_models_use_max_completion_tokens": True,
            }
        ),
        "azure_openai": MappingProxyType(
            {
                "supports_response_format": True,
                "supports_streaming": True,
                "supports_tools": True,
                "system_in_messages": True,
                "newer_models_use_max_completion_tokens": True,
                "requires_api_version": True,
            }
        ),
        "anthropic": MappingProxyType(
            {
                "supports_response_format": False,
                "supports_streaming": True,
                "supports_tools": True,
                "system_in_messages": False,
                "has_thinking_tags": False,
            }
        ),
        "claude": MappingProxyType(
            {  # Legacy alias for Anthropic
                "supports_response_format": False,
                "supports_streaming": True,
                "supports_tools": True,
                "system_in_messages": False,
                "has_thinking_tags": False,
            }
        ),
        "deepseek": MappingProxyType(
            {
                "supports_response_format": True,
                "supports_streaming": True,
                "supports_tools": True,
                "system_in_messages": True,
                "has_thinking_tags": True,
                "thinking_markers": MappingProxyType(
                    {"open": ("<think>", "◣"), "close": ("</think>", "◢")}
                ),
            }
        ),
        "ollama": MappingProxyType(
            {
                "supports_response_format": True,
                "supports_streaming": True,
                "supports_tools": True,
                "system_in_messages": True,
            }
        ),
        "openrouter": MappingProxyType(
            {
                "supports_response_format": True,
                "supports_streaming": True,
                "supports_tools": True,
                "system_in_messages": True,
            }
        ),
    }
)

DEFAULT_CAPABILITIES: Mapping[str, Any] = MappingProxyType(
    {
        "supports_response_format": True,
        "supports_streaming": True,
        "supports_tools": False,
        "system_in_messages": True,
        "has_thinking_tags": False,
        "forced_temperature": None,
    }
)

MODEL_OVERRIDES: Mapping[str, Mapping[str, Any]] = MappingProxyType(
    {
        "deepseek-reasoner": MappingProxyType(
            {
                "supports_response_format": False,
                "has_thinking_tags": True,
            }
        ),
        "qwen": MappingProxyType(
            {
                "has_thinking_tags": True,
                "thinking_markers": MappingProxyType(
                    {"open": ("<think>", "◣"), "close": ("</think>", "◢")}
                ),
            }
        ),
        "o1": MappingProxyType({"forced_temperature": 1.0}),
        "o3": MappingProxyType({"forced_temperature": 1.0}),
    }
)

_SORTED_OVERRIDE_PATTERNS: Tuple[Tuple[str, Mapping[str, Any]], ...] = tuple(
    sorted(MODEL_OVERRIDES.items(), key=lambda x: -len(x[0]))
)

_RUNTIME_CAPABILITIES: dict[str, Mapping[str, Any]] = {}
_RUNTIME_LOCK = threading.RLock()


@functools.lru_cache(maxsize=1)
def _log_missing_binding_warning() -> None:
    """
    Thread-safe, idempotent warning for deprecated API usage.
    Uses lru_cache to ensure the log is emitted exactly once per process runtime.
    """
    warnings.warn(
        "Deprecated API usage: get_capability called without binding. "
        f"Fallback to '{DEFAULT_PROVIDER}' used.",
        DeprecationWarning,
        stacklevel=2,
    )


def _resolve_model_override(model: Optional[str]) -> Mapping[str, Any]:
    """
    Centralized logic to resolve model-specific overrides.
    Returns an empty mapping if no override matches.
    """
    if not model:
        return MappingProxyType({})

    m_lower = model.lower()
    for pattern, overrides in _SORTED_OVERRIDE_PATTERNS:
        if m_lower.startswith(pattern):
            return overrides
    return MappingProxyType({})


def register_provider_capabilities(binding: str, capabilities: Mapping[str, Any]) -> None:
    """Register runtime capability overrides for a provider.

    Args:
        binding: Provider binding name.
        capabilities: Capability mapping to override defaults.

    Returns:
        None.

    Raises:
        None.
    """
    binding_key = binding.lower().strip()
    with _RUNTIME_LOCK:
        _RUNTIME_CAPABILITIES[binding_key] = MappingProxyType(dict(capabilities))


def _get_runtime_capabilities(binding_key: str) -> Mapping[str, Any] | None:
    with _RUNTIME_LOCK:
        return _RUNTIME_CAPABILITIES.get(binding_key)


def get_capability(
    binding: Optional[str],
    capability: str,
    model: Optional[str] = None,
    default: Any = None,
) -> Any:
    """
    Retrieve capability value using hierarchical resolution.

    Resolution order:
    1. Model-specific overrides (Prefix matched).
    2. Runtime registry overrides.
    3. Provider-specific capabilities.
    4. Registry defaults.
    """
    if not binding:
        _log_missing_binding_warning()

    binding_key = str(binding or DEFAULT_PROVIDER).lower()

    overrides = _resolve_model_override(model)
    if capability in overrides:
        return overrides[capability]

    runtime_caps = _get_runtime_capabilities(binding_key)
    if runtime_caps and capability in runtime_caps:
        return runtime_caps[capability]

    provider_caps = PROVIDER_CAPABILITIES.get(binding_key)
    if provider_caps and capability in provider_caps:
        return provider_caps[capability]

    return DEFAULT_CAPABILITIES.get(capability, default)


def get_thinking_markers(
    binding: Optional[str], model: Optional[str] = None
) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """Resolve open/close thinking markers for a given provider/model."""
    binding_key = str(binding or DEFAULT_PROVIDER).lower()

    overrides = _resolve_model_override(model)
    if "thinking_markers" in overrides:
        markers = overrides["thinking_markers"]
        return tuple(markers.get("open", ())), tuple(markers.get("close", ()))

    runtime_caps = _get_runtime_capabilities(binding_key)
    if runtime_caps and "thinking_markers" in runtime_caps:
        markers = runtime_caps["thinking_markers"]
        return tuple(markers.get("open", ())), tuple(markers.get("close", ()))

    provider_caps = PROVIDER_CAPABILITIES.get(binding_key, {})
    if "thinking_markers" in provider_caps:
        markers = provider_caps["thinking_markers"]
        return tuple(markers.get("open", ())), tuple(markers.get("close", ()))

    if get_capability(binding, "has_thinking_tags", model):
        return DEFAULT_OPEN_MARKERS, DEFAULT_CLOSE_MARKERS

    return (), ()


def supports_response_format(binding: Optional[str], model: Optional[str] = None) -> bool:
    """Check if the provider/model supports response_format JSON mode."""
    return bool(get_capability(binding, "supports_response_format", model, default=True))


def supports_streaming(binding: Optional[str], model: Optional[str] = None) -> bool:
    """Check if the provider/model supports streaming responses."""
    return bool(get_capability(binding, "supports_streaming", model, default=True))


def system_in_messages(binding: Optional[str], model: Optional[str] = None) -> bool:
    """Check if the system prompt should be placed in messages array (OpenAI style)."""
    return bool(get_capability(binding, "system_in_messages", model, default=True))


def has_thinking_tags(binding: Optional[str], model: Optional[str] = None) -> bool:
    """Check if output likely contains reasoning/thinking blocks."""
    return bool(get_capability(binding, "has_thinking_tags", model, default=False))


def supports_tools(binding: Optional[str], model: Optional[str] = None) -> bool:
    """Check if the provider supports tool/function calling."""
    return bool(get_capability(binding, "supports_tools", model, default=False))


def requires_api_version(binding: Optional[str], model: Optional[str] = None) -> bool:
    """Check if the provider requires an api_version parameter (e.g., Azure OpenAI)."""
    return bool(get_capability(binding, "requires_api_version", model, default=False))


def uses_completion_tokens(binding: Optional[str], model: Optional[str] = None) -> bool:
    """Check whether the provider/model uses max_completion_tokens vs max_tokens."""
    return bool(
        get_capability(binding, "newer_models_use_max_completion_tokens", model, default=False)
    )


def get_effective_temperature(
    binding: Optional[str], model: Optional[str] = None, requested: float = 0.7
) -> float:
    """Return forced temperature for models with restricted parameters."""
    forced = get_capability(binding, "forced_temperature", model)
    return float(forced) if forced is not None else requested


__all__ = [
    "get_capability",
    "register_provider_capabilities",
    "supports_response_format",
    "supports_streaming",
    "system_in_messages",
    "has_thinking_tags",
    "supports_tools",
    "requires_api_version",
    "uses_completion_tokens",
    "get_effective_temperature",
]
