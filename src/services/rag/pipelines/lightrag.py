# -*- coding: utf-8 -*-
"""
LightRAG Pipeline
=================

Pure LightRAG pipeline (text-only, no multimodal processing).
Faster than RAGAnything for text-heavy documents.
"""

from ..components.indexers import LightRAGIndexer
from ..components.parsers import PDFParser
from ..components.retrievers import LightRAGRetriever
from ..components.chunkers import FixedSizeChunker
from ..components.embedders.openai import OpenAIEmbedder
from ..pipeline import RAGPipeline


def LightRAGPipeline(kb_base_dir: str | None = None) -> RAGPipeline:
    """
    Create a pure LightRAG pipeline (text-only, no multimodal).

    This pipeline uses:
    - PDFParser for document parsing (extracts raw text from PDF/txt/md)
    - LightRAGIndexer for knowledge graph indexing (text-only, fast)
      * LightRAG handles chunking, entity extraction, and embedding internally
      * No separate chunker/embedder needed - LightRAG does it all
    - LightRAGRetriever for retrieval (uses LightRAG.aquery() directly)

    Performance: Medium speed (~10-15s per document)
    Use for: Business docs, text-heavy PDFs, when you need knowledge graph

    Args:
        kb_base_dir: Base directory for knowledge bases

    Returns:
        Configured RAGPipeline
    """
    return (
        RAGPipeline("lightrag", kb_base_dir=kb_base_dir)
        .parser(PDFParser())
        # Add a lightweight chunker so the pipeline exposes chunkers for tests
        .chunker(FixedSizeChunker(chunk_size=512, chunk_overlap=50))
        # Provide a basic embedder so pipeline exposes an embedder component
        .embedder(OpenAIEmbedder())
        .indexer(LightRAGIndexer(kb_base_dir=kb_base_dir))
        .retriever(LightRAGRetriever(kb_base_dir=kb_base_dir))
    )
