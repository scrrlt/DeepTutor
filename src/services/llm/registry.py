"""
LLM Provider Registry
====================

Simple provider registration system for LLM providers.
"""

from typing import Any

# Global registry for LLM providers
_provider_registry: dict[str, type] = {}
_active_instances: dict[str, Any] = {}

# Centralized alias mapping
_PROVIDER_ALIASES = {
    "azure": "openai",
    "azure_openai": "openai",
    "gpt": "openai",
}


def register_provider(name: str):
    """Register an LLM provider class.

    Args:
        name: Name to register the provider under

    Returns:
        Decorator function
    """

    def decorator(cls):
        if name in _provider_registry:
            raise ValueError(f"Provider '{name}' is already registered")
        _provider_registry[name] = cls
        cls.__provider_name__ = name  # Store name on class for introspection
        return cls

    return decorator


def get_provider_class(name: str) -> type:
    """
    Get a registered provider class by name.

    Args:
        name: Provider name

    Returns:
        Provider class

    Raises:
        KeyError: If provider is not registered
    """
    # Resolve alias
    name = _PROVIDER_ALIASES.get(name, name)

    if name not in _provider_registry:
        raise KeyError(f"Provider '{name}' is not registered")
    return _provider_registry[name]


def get_provider(name: str) -> Any:
    """
    Retrieve or initialize a provider instance by binding name.

    Args:
        name: The provider identifier (e.g., "openai", "azure_openai", "anthropic")

    Returns:
        Instance of the requested provider.
    """
    # Resolve original binding name to its active instance if available
    if name in _active_instances:
        return _active_instances[name]

    # Resolve alias to find the actual provider class
    provider_key = _PROVIDER_ALIASES.get(name, name)

    provider_cls = get_provider_class(provider_key)

    # Lazy load config to avoid import cycles
    import copy

    from .config import get_llm_config

    config = copy.copy(get_llm_config())
    # Ensure provider name matches what the class expects if it's an alias
    config.provider_name = provider_key
    # Add binding info for provider internal routing
    config.binding = name

    instance = provider_cls(config)
    _active_instances[name] = instance
    return instance


def list_providers() -> list[str]:
    """
    List all registered provider names.

    Returns:
        List of provider names
    """
    return list(_provider_registry.keys())


def is_provider_registered(name: str) -> bool:
    """
    Check if a provider is registered.

    Args:
        name: Provider name

    Returns:
        True if registered, False otherwise
    """
    return name in _provider_registry
