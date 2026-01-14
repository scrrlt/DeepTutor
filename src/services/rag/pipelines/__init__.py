"""
Pre-configured Pipelines
========================

Ready-to-use RAG pipelines for common use cases.
"""

# Avoid top-level imports of pipelines to prevent eager loading of heavy dependencies
# Instead, the factory will import them lazily as needed.


__all__ = [
    "RAGAnythingPipeline",
    "LightRAGPipeline",
    "LlamaIndexPipeline",
    "AcademicPipeline",
]
