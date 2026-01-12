"""
Pre-configured Pipelines
========================

Ready-to-use RAG pipelines for common use cases.
"""

from .academic import AcademicPipeline
from .lightrag import LightRAGPipeline
from .raganything import RAGAnythingPipeline

# Conditionally import LlamaIndex pipeline if available
try:
    from .llamaindex import LlamaIndexPipeline

    _llamaindex_available = True
except ImportError:
    _llamaindex_available = False
    LlamaIndexPipeline = None

__all__ = [
    "RAGAnythingPipeline",
    "LightRAGPipeline",
    "LlamaIndexPipeline",
    "AcademicPipeline",
]
