"""
Pre-configured Pipelines
========================

Ready-to-use RAG pipelines for common use cases.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "RAGAnythingPipeline",
    "RAGAnythingDoclingPipeline",
    "LightRAGPipeline",
    "LlamaIndexPipeline",
]

# NOTE:
# - Do NOT import heavy/optional backends at module import time.
# - Users may want `llamaindex` without `raganything`, or vice versa.
# - Accessing an attribute triggers a targeted import via __getattr__.


def __getattr__(name: str) -> Any:
    if name == "LightRAGPipeline":
        from .lightrag import LightRAGPipeline

        return LightRAGPipeline
    if name == "RAGAnythingPipeline":
        from .raganything import RAGAnythingPipeline

        return RAGAnythingPipeline
    if name == "RAGAnythingDoclingPipeline":
        from .raganything_docling import RAGAnythingDoclingPipeline

        return RAGAnythingDoclingPipeline
    if name == "LlamaIndexPipeline":
        from .llamaindex import LlamaIndexPipeline

        return LlamaIndexPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
