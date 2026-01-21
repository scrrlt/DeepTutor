# -*- coding: utf-8 -*-
"""
Provider Capabilities
=====================

Centralized configuration for LLM provider capabilities.
Optimized for performance via pre-sorted lookups and enforced immutability.
"""

import logging
from types import MappingProxyType
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Module-level state for tracking missing bindings (rate-limited)
_warned_missing_binding = False

# Standard thinking markers.
DEFAULT_OPEN_MARKERS: Tuple[str, ...] = ("<think>",)
DEFAULT_CLOSE_MARKERS: Tuple[str, ...] = ("</think>",)

# Frozen provider capabilities.
# Using MappingProxyType ensures read-only integrity across the system.
PROVIDER_CAPABILITIES: Dict[str, Any] = MappingProxyType(
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
                "supports_response_format": True,  # V3 supports JSON mode
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
                "supports_tools": True,  # Ollama supports function calling
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

# Frozen default fallbacks
DEFAULT_CAPABILITIES: Dict[str, Any] = MappingProxyType(
    {
        "supports_response_format": True,
        "supports_streaming": True,
        "supports_tools": False,
        "system_in_messages": True,
        "has_thinking_tags": False,
        "forced_temperature": None,
    }
)

# Model-specific overrides.
MODEL_OVERRIDES: Dict[str, Dict[str, Any]] = MappingProxyType(
    {
        "deepseek-reasoner": MappingProxyType(
            {
                "supports_response_format": False,  # R1 struggles with strict JSON schemas
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

# Performance Optimization: Pre-sorted override patterns for O(N) lookup.
_SORTED_OVERRIDE_PATTERNS: Tuple[Tuple[str, Dict[str, Any]], ...] = tuple(
    sorted(MODEL_OVERRIDES.items(), key=lambda x: -len(x[0]))
)

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
    2. Provider-specific capabilities.
    3. Registry defaults.

    Returns the raw value or a MappingProxy. Mutation is explicitly disallowed.
    """
    global _warned_missing_binding
    if not binding and not _warned_missing_binding:
        # Fallback to OpenAI is deprecated; will be enforced as a hard error in future releases.
        logger.warning(
            "Deprecated API usage: get_capability called without binding. Fallback to 'openai' used."
        )
        _warned_missing_binding = True

    binding_lower = (binding or "openai").lower()

    # 1. Check pre-sorted model overrides
    if model:
        m_lower = model.lower()
        for pattern, overrides in _SORTED_OVERRIDE_PATTERNS:
            if m_lower.startswith(pattern):
                if capability in overrides:
                    return overrides[capability]

    # 2. Check provider capabilities
    provider_caps = PROVIDER_CAPABILITIES.get(binding_lower, {})
    if capability in provider_caps:
        return provider_caps[capability]

    # 3. Final default fallback
    return DEFAULT_CAPABILITIES.get(capability, default)


def get_thinking_markers(
    binding: Optional[str], model: Optional[str] = None
) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """Resolve open/close thinking markers for a given provider/model."""
    if model:
        m_lower = model.lower()
        for pattern, overrides in _SORTED_OVERRIDE_PATTERNS:
            if m_lower.startswith(pattern):
                markers = overrides.get("thinking_markers")
                if markers:
                    return tuple(markers.get("open", ())), tuple(markers.get("close", ()))

    binding_key = (binding or "openai").lower()
    provider_caps = PROVIDER_CAPABILITIES.get(binding_key, {})
    markers = provider_caps.get("thinking_markers")
    if markers:
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
    "supports_response_format",
    "supports_streaming",
    "system_in_messages",
    "has_thinking_tags",
    "supports_tools",
    "requires_api_version",
    "uses_completion_tokens",
    "get_effective_temperature",
    "get_thinking_markers",
]
