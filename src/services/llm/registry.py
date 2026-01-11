"""
Auto-Registry Pattern - Stop touching factory.py.
"""

from typing import Dict, Type

_PROVIDERS: Dict[str, Type] = {}


def register_provider(name: str):
    """Decorator to register a provider class."""
    def decorator(cls):
        _PROVIDERS[name] = cls
        return cls
    return decorator


def get_provider_class(name: str) -> Type:
    """Get registered provider class by name."""
    return _PROVIDERS.get(name)
