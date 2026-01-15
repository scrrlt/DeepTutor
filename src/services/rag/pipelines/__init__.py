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
    "LightRAGPipeline",
    "LlamaIndexPipeline",
    "AcademicPipeline",
]
