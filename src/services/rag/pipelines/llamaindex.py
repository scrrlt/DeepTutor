"""
LlamaIndex Pipeline
===================

True LlamaIndex integration using official llama-index library.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import asyncio

from src.logging import get_logger

try:
    from llama_index.core import (
        Document,
        Settings,
        StorageContext,
        VectorStoreIndex,
        load_index_from_storage,
    )
    from llama_index.core.base.embeddings.base import BaseEmbedding
    from llama_index.core.bridge.pydantic import PrivateAttr
    
    import fitz  # PyMuPDF
    import nest_asyncio
    
    LLAMAINDEX_AVAILABLE = True

except ImportError:
    LLAMAINDEX_AVAILABLE = False


# Default knowledge base directory
DEFAULT_KB_BASE_DIR = str(
    Path(__file__).resolve().parent.parent.parent.parent.parent / "data" / "knowledge_bases"
)


if LLAMAINDEX_AVAILABLE:

    class CustomEmbedding(BaseEmbedding):
        """
        Custom embedding adapter for OpenAI-compatible APIs.
        
        Works with any OpenAI-compatible endpoint including:
        - Google Gemini (text-embedding-004)
        - OpenAI (text-embedding-ada-002, text-embedding-3-*)
        - Azure OpenAI
        - Local models with OpenAI-compatible API
        """
        
        _client: Any = PrivateAttr()
        
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            # Conditional import
            from src.services.embedding import get_embedding_client
            self._client = get_embedding_client()
            
        @classmethod
        def class_name(cls) -> str:
            return "custom_embedding"
        
        async def _aget_query_embedding(self, query: str) -> List[float]:
            """Get embedding for a query."""
            embeddings = await self._client.embed([query])
            return embeddings[0]
        
        async def _aget_text_embedding(self, text: str) -> List[float]:
            """Get embedding for a text."""
            embeddings = await self._client.embed([text])
            return embeddings[0]
        
        def _get_query_embedding(self, query: str) -> List[float]:
            """Sync version - called by LlamaIndex sync API."""
            nest_asyncio.apply()
            return asyncio.run(self._aget_query_embedding(query))
        
        def _get_text_embedding(self, text: str) -> List[float]:
            """Sync version - called by LlamaIndex sync API."""
            nest_asyncio.apply()
            return asyncio.run(self._aget_text_embedding(text))
        
        async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
            """Get embeddings for multiple texts."""
            return await self._client.embed(texts)


    class LlamaIndexPipeline:
        """
        True LlamaIndex pipeline using official llama-index library.
        
        Uses LlamaIndex's native components:
        - VectorStoreIndex for indexing
        - CustomEmbedding for OpenAI-compatible embeddings
        - SentenceSplitter for chunking
        - StorageContext for persistence
        """

        def __init__(self, kb_base_dir: Optional[str] = None):
            """
            Initialize LlamaIndex pipeline.

            Args:
                kb_base_dir: Base directory for knowledge bases
            """
            self.logger = get_logger("LlamaIndexPipeline")
            self.kb_base_dir = kb_base_dir or DEFAULT_KB_BASE_DIR
            self._configure_settings()

        def _configure_settings(self):
            """Configure LlamaIndex global settings."""
            from src.services.embedding import get_embedding_config
            
            embedding_cfg = get_embedding_config()
            Settings.embed_model = CustomEmbedding()
            Settings.chunk_size = 512
            Settings.chunk_overlap = 50
            
            self.logger.info(
                f"LlamaIndex configured: embedding={embedding_cfg.model} "
                f"({embedding_cfg.dim}D, {embedding_cfg.binding}), chunk_size=512"
            )

        async def initialize(
            self, kb_name: str, file_paths: List[str], **kwargs
        ) -> bool:
            self.logger.info(f"Initializing KB '{kb_name}' with {len(file_paths)} files using LlamaIndex")

            kb_dir = Path(self.kb_base_dir) / kb_name
            storage_dir = kb_dir / "llamaindex_storage"
            storage_dir.mkdir(parents=True, exist_ok=True)

            try:
                documents = []
                for file_path in file_paths:
                    file_path = Path(file_path)
                    self.logger.info(f"Parsing: {file_path.name}")
                    
                    if file_path.suffix.lower() == ".pdf":
                        text = self._extract_pdf_text(file_path)
                    else:
                        with open(file_path, "r", encoding="utf-8") as f:
                            text = f.read()
                    
                    if text.strip():
                        doc = Document(
                            text=text,
                            metadata={"file_name": file_path.name, "file_path": str(file_path)}
                        )
                        documents.append(doc)
                        self.logger.info(f"Loaded: {file_path.name} ({len(text)} chars)")
                    else:
                        self.logger.warning(f"Skipped empty document: {file_path.name}")

                if not documents:
                    self.logger.error("No valid documents found")
                    return False

                self.logger.info(f"Creating VectorStoreIndex with {len(documents)} documents...")
                
                loop = asyncio.get_event_loop()
                index = await loop.run_in_executor(
                    None, lambda: VectorStoreIndex.from_documents(documents, show_progress=True)
                )

                index.storage_context.persist(persist_dir=str(storage_dir))
                self.logger.info(f"Index persisted to {storage_dir}")
                self.logger.info(f"KB '{kb_name}' initialized successfully with LlamaIndex")
                return True

            except Exception as e:
                self.logger.error(f"Failed to initialize KB: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
                return False

        def _extract_pdf_text(self, file_path: Path) -> str:
            try:
                doc = fitz.open(file_path)
                texts = [page.get_text() for page in doc]
                doc.close()
                return "\n\n".join(texts)
            except Exception as e:
                self.logger.error(f"Failed to extract PDF text: {e}")
                return ""

        async def search(
            self, query: str, kb_name: str, mode: str = "hybrid", **kwargs
        ) -> Dict[str, Any]:
            self.logger.info(f"Searching KB '{kb_name}' with query: {query[:50]}...")

            kb_dir = Path(self.kb_base_dir) / kb_name
            storage_dir = kb_dir / "llamaindex_storage"

            if not storage_dir.exists():
                return {"query": query, "answer": "No documents indexed.", "content": ""}

            try:
                loop = asyncio.get_event_loop()
                
                def load_and_retrieve():
                    storage_context = StorageContext.from_defaults(persist_dir=str(storage_dir))
                    index = load_index_from_storage(storage_context)
                    retriever = index.as_retriever(similarity_top_k=kwargs.get("top_k", 5))
                    return retriever.retrieve(query)
                
                nodes = await loop.run_in_executor(None, load_and_retrieve)
                content = "\n\n".join([node.node.text for node in nodes])

                return {"query": query, "answer": content, "content": content, "mode": mode, "provider": "llamaindex"}

            except Exception as e:
                self.logger.error(f"Search failed: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
                return {"query": query, "answer": f"Search failed: {e}", "content": ""}

        async def delete(self, kb_name: str) -> bool:
            import shutil
            kb_dir = Path(self.kb_base_dir) / kb_name
            if kb_dir.exists():
                shutil.rmtree(kb_dir)
                self.logger.info(f"Deleted KB '{kb_name}'")
                return True
            return False

else:
    # ==================================================================
    # Dummy class if LlamaIndex dependencies are not installed
    # ==================================================================
    class LlamaIndexPipeline:
        def __init__(self, kb_base_dir: Optional[str] = None):
            self.logger = get_logger("LlamaIndexPipeline")
            self.logger.warning(
                "LlamaIndex dependencies not found. Running in dummy mode. "
                "Please install with 'pip install llama-index llama-parse PyMuPDF'."
            )

        async def initialize(self, kb_name: str, file_paths: List[str], **kwargs) -> bool:
            self.logger.warning("LlamaIndex not available. Initialization skipped.")
            return False

        async def search(self, query: str, kb_name: str, **kwargs) -> Dict[str, Any]:
            self.logger.warning("LlamaIndex not available. Search skipped.")
            return {
                "query": query,
                "answer": "Search is disabled because LlamaIndex dependencies are not installed.",
                "content": "",
            }

        async def delete(self, kb_name: str) -> bool:
            self.logger.warning("LlamaIndex not available. Deletion skipped.")
            return False

