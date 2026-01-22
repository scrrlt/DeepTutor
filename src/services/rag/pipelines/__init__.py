# -*- coding: utf-8 -*-
"""
Pre-configured Pipelines
========================

Ready-to-use RAG pipelines for common use cases.
"""

from .academic import AcademicPipeline
from .lightrag import LightRAGPipeline

try:
    from .llamaindex import LlamaIndexPipeline
except ImportError as e:
    _llamaindex_import_error = e

    class LlamaIndexPipeline:
        def __init__(self, *args: object, **kwargs: object) -> None:
            raise ImportError(
                "LlamaIndex not installed. Install with: pip install llama-index"
            ) from _llamaindex_import_error


from .raganything import RAGAnythingPipeline

__all__ = [
    "RAGAnythingPipeline",
    "RAGAnythingDoclingPipeline",
    "LightRAGPipeline",
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
        # Optional dependency: llama_index
        from .llamaindex import LlamaIndexPipeline

        return LlamaIndexPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
