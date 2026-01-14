from typing import Type, Dict, Optional
from .provider import BaseLLMProvider

# The global registry dictionary
_PROVIDER_REGISTRY: Dict[str, Type[BaseLLMProvider]] = {}

def register_provider(name: str):
    """
    Decorator to register a provider class.
    Usage: @register_provider("openai")
    """
    def decorator(cls: Type[BaseLLMProvider]):
        _PROVIDER_REGISTRY[name.lower()] = cls
        cls.provider_name = name.lower()
        return cls
    return decorator

def get_provider_class(name: str) -> Optional[Type[BaseLLMProvider]]:
    """Retrieves a provider class by name (case-insensitive)."""
    return _PROVIDER_REGISTRY.get(name.lower())

def list_providers() -> list[str]:
    return list(_PROVIDER_REGISTRY.keys())
