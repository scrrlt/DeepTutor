"""Pure LightRAG indexer for text-only processing."""

import asyncio
from pathlib import Path
import sys
from typing import Any, ClassVar

from src.services.llm.cache import get_cache_client

from ...types import Document
from ..base import BaseComponent


class LightRAGIndexer(BaseComponent):
    """
    Pure LightRAG knowledge graph indexer (text-only).

    Uses LightRAG library directly without multimodal processing.
    Faster than RAGAnything for text-only documents.
    """

    name = "lightrag_indexer"
    _instances: ClassVar[dict[str, Any]] = {}  # Cache LightRAG instances
    _init_locks: ClassVar[dict[str, asyncio.Lock]] = {}

    def __init__(self, kb_base_dir: str | None = None):
        """
        Initialize LightRAG indexer.

        Args:
            kb_base_dir: Base directory for knowledge bases
        """
        super().__init__()
        self.kb_base_dir = kb_base_dir or str(
            Path(__file__).resolve().parent.parent.parent.parent.parent.parent
            / "data"
            / "knowledge_bases"
        )

    async def _get_lightrag_instance(self, kb_name: str):
        """Get or create a LightRAG instance (text-only)."""
        working_dir = str(Path(self.kb_base_dir) / kb_name / "rag_storage")

        if working_dir in self._instances:
            return self._instances[working_dir]

        local_lock = self._init_locks.setdefault(working_dir, asyncio.Lock())

        async with local_lock:
            if working_dir in self._instances:
                return self._instances[working_dir]

            project_root = (
                Path(__file__).resolve().parent.parent.parent.parent.parent.parent
            )
            raganything_path = project_root.parent / "raganything" / "RAG-Anything"
            if raganything_path.exists() and str(raganything_path) not in sys.path:
                sys.path.insert(0, str(raganything_path))

            try:
                from lightrag import LightRAG

                from src.services.embedding import get_embedding_client
                from src.services.llm import get_llm_client

                llm_client = get_llm_client()
                embed_client = get_embedding_client()

                distributed_lock = None
                acquired = False
                cache_client = await get_cache_client()
                if cache_client:
                    distributed_lock = cache_client.lock(
                        f"lightrag:init:{working_dir}",
                        timeout=120,
                        blocking_timeout=30,
                    )
                    acquired = await distributed_lock.acquire()

                try:
                    llm_model_func = llm_client.get_model_func()

                    rag = LightRAG(
                        working_dir=working_dir,
                        llm_model_func=llm_model_func,
                        embedding_func=embed_client.get_embedding_func(),
                    )

                    self._instances[working_dir] = rag
                    return rag
                finally:
                    if distributed_lock and acquired:
                        try:
                            await distributed_lock.release()
                        except Exception as exc:  # noqa: BLE001
                            self.logger.warning(
                                "Failed to release LightRAG init lock: %s",
                                exc,
                            )

            except ImportError as exc:
                self.logger.error("Failed to import LightRAG: %s", exc)
                raise

    async def process(
        self,
        kb_name: str,
        documents: list[Document],
        **kwargs: object,
    ) -> bool:
        """
        Build knowledge graph from documents (text-only).

        Args:
            kb_name: Knowledge base name
            documents: List of documents to index
            **kwargs: Additional arguments

        Returns:
            True if successful
        """
        self.logger.info(f"Building knowledge graph for {kb_name} (text-only)...")

        from src.logging.adapters import LightRAGLogContext

        # Use log forwarding context
        with LightRAGLogContext(scene="LightRAG-Indexer"):
            rag = await self._get_lightrag_instance(kb_name)

            # Initialize storages (required for LightRAG)
            await rag.initialize_storages()

            # Initialize pipeline status (required for document processing)
            from lightrag.kg.shared_storage import initialize_pipeline_status

            await initialize_pipeline_status()

            for doc in documents:
                if doc.content:
                    # Use direct LightRAG insert (text-only, fast)
                    await rag.ainsert(doc.content)

        self.logger.info("Knowledge graph built successfully (text-only)")
        return True
