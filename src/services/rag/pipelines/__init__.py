"""
Pre-configured Pipelines
========================

Ready-to-use RAG pipelines for common use cases.
"""

from .raganything import RAGAnythingPipeline
from .lightrag import LightRAGPipeline
from .llamaindex import LlamaIndexPipeline
from .academic import AcademicPipeline

__all__ = [
    "RAGAnythingPipeline",
    "LightRAGPipeline",
    "LlamaIndexPipeline",
    "AcademicPipeline",
]

