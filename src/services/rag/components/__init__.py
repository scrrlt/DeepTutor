"""
RAG Components
==============

Modular components for building RAG pipelines.

Components follow a simple protocol:
- Each component has a `name` attribute
- Each component has an async `process()` method
"""

from .base import Component, BaseComponent

# Import component modules for convenience
from . import parsers
from . import chunkers
from . import embedders
from . import indexers
from . import retrievers

__all__ = [
    "Component",
    "BaseComponent",
    "parsers",
    "chunkers",
    "embedders",
    "indexers",
    "retrievers",
]

