"""
Base Chunker
============

Base class for document chunkers.
"""

from typing import List

from ..base import BaseComponent
from ...types import Document, Chunk


class BaseChunker(BaseComponent):
    """
    Base class for document chunkers.
    
    Chunkers split documents into smaller chunks for processing.
    """
    
    name = "base_chunker"

    async def process(self, doc: Document, **kwargs) -> List[Chunk]:
        """
        Chunk a document.
        
        Args:
            doc: Document to chunk
            **kwargs: Additional arguments
            
        Returns:
            List of Chunks
        """
        raise NotImplementedError("Subclasses must implement process()")

