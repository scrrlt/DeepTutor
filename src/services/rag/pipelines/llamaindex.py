"""
LlamaIndex Pipeline
===================

Component-based pipeline using vector indexing for fast retrieval.
"""

from ..pipeline import RAGPipeline
from ..components.parsers import PDFParser
from ..components.chunkers import SemanticChunker
from ..components.embedders import OpenAIEmbedder
from ..components.indexers import VectorIndexer
from ..components.retrievers import DenseRetriever


def LlamaIndexPipeline() -> RAGPipeline:
    """
    Create a LlamaIndex-style pipeline.
    
    This pipeline uses:
    - PDFParser for document parsing
    - SemanticChunker for text chunking
    - OpenAIEmbedder for embedding generation
    - VectorIndexer for vector indexing
    - DenseRetriever for dense retrieval
    
    Returns:
        Configured RAGPipeline
    """
    return (
        RAGPipeline("llamaindex")
        .parser(PDFParser())
        .chunker(SemanticChunker())
        .embedder(OpenAIEmbedder())
        .indexer(VectorIndexer())
        .retriever(DenseRetriever())
    )

