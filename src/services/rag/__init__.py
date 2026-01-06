"""
RAG Service
===========

Unified RAG pipeline service for DeepTutor.

Provides:
- Composable RAG pipelines
- Pre-configured pipelines (RAGAnything, LightRAG, LlamaIndex, Academic)
- Modular components (parsers, chunkers, embedders, indexers, retrievers)
- Factory for pipeline creation

Usage:
    from src.services.rag import get_pipeline, RAGPipeline, RAGAnythingPipeline

    # Get pre-configured pipeline
    pipeline = get_pipeline("raganything")
    await pipeline.initialize("kb_name", ["doc1.pdf", "doc2.pdf"])
    result = await pipeline.search("query", "kb_name")

    # Or build custom pipeline
    from src.services.rag.components import PDFParser, SemanticChunker, GraphIndexer

    custom = (
        RAGPipeline("custom")
        .parser(PDFParser())
        .chunker(SemanticChunker())
        .indexer(GraphIndexer())
    )
"""

from .types import Document, Chunk
from .pipeline import RAGPipeline
from .factory import get_pipeline, list_pipelines, register_pipeline

# Import pipeline classes for convenience
from .pipelines.raganything import RAGAnythingPipeline

__all__ = [
    # Types
    "Document",
    "Chunk",
    # Pipeline
    "RAGPipeline",
    # Factory
    "get_pipeline",
    "list_pipelines",
    "register_pipeline",
    # Pipeline implementations
    "RAGAnythingPipeline",
]

