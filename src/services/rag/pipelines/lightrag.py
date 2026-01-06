"""
LightRAG Pipeline
=================

Component-based pipeline using LightRAG for knowledge graph indexing.
"""

from ..pipeline import RAGPipeline
from ..components.parsers import PDFParser
from ..components.chunkers import SemanticChunker
from ..components.embedders import OpenAIEmbedder
from ..components.indexers import GraphIndexer
from ..components.retrievers import HybridRetriever


def LightRAGPipeline() -> RAGPipeline:
    """
    Create a LightRAG-based pipeline.
    
    This pipeline uses:
    - PDFParser for document parsing
    - SemanticChunker for text chunking
    - OpenAIEmbedder for embedding generation
    - GraphIndexer for knowledge graph indexing
    - HybridRetriever for retrieval
    
    Returns:
        Configured RAGPipeline
    """
    return (
        RAGPipeline("lightrag")
        .parser(PDFParser())
        .chunker(SemanticChunker())
        .embedder(OpenAIEmbedder())
        .indexer(GraphIndexer())
        .retriever(HybridRetriever())
    )

