# -*- coding: utf-8 -*-
"""
LightRAG Pipeline
=================

Pure LightRAG pipeline (text-only, no multimodal processing).
Faster than RAGAnything for text-heavy documents.
"""

from typing import Optional

from ..components.indexers import LightRAGIndexer
from ..components.parsers import PDFParser
from ..components.retrievers import LightRAGRetriever
from ..components.chunkers import FixedSizeChunker
from ..components.embedders.openai import OpenAIEmbedder
from ..pipeline import RAGPipeline


def LightRAGPipeline(kb_base_dir: Optional[str] = None) -> RAGPipeline:
    """
    Create and configure a LightRAG RAGPipeline for text-heavy, text-only documents.
    
    The pipeline is configured with a PDFParser, a FixedSizeChunker (chunk_size=512, chunk_overlap=50),
    an OpenAIEmbedder, a LightRAGIndexer, and a LightRAGRetriever to provide knowledge-graph backed retrieval.
    
    Parameters:
        kb_base_dir (Optional[str]): Base directory for knowledge bases; passed to the indexer and retriever.
    
    Returns:
        RAGPipeline: A configured RAGPipeline instance named "lightrag".
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