"""
Academic Pipeline
=================

Pipeline optimized for academic documents with numbered item extraction.
"""

from ..pipeline import RAGPipeline
from ..components.parsers import PDFParser
from ..components.chunkers import SemanticChunker, NumberedItemExtractor
from ..components.embedders import OpenAIEmbedder
from ..components.indexers import GraphIndexer
from ..components.retrievers import HybridRetriever


def AcademicPipeline() -> RAGPipeline:
    """
    Create an academic document pipeline.
    
    This pipeline uses:
    - PDFParser with MinerU for multimodal parsing
    - SemanticChunker for text chunking
    - NumberedItemExtractor for extracting definitions, theorems, etc.
    - OpenAIEmbedder for embedding generation
    - GraphIndexer for knowledge graph indexing
    - HybridRetriever for hybrid retrieval
    
    Returns:
        Configured RAGPipeline
    """
    return (
        RAGPipeline("academic")
        .parser(PDFParser(use_mineru=True))
        .chunker(SemanticChunker())
        .chunker(NumberedItemExtractor())
        .embedder(OpenAIEmbedder())
        .indexer(GraphIndexer())
        .retriever(HybridRetriever())
    )

