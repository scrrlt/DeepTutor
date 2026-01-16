"""Pure LightRAG retriever for text-only queries."""

import asyncio

from pathlib import Path
import sys
from typing import Any, ClassVar, Dict, Optional

from ..base import BaseComponent
from src.services.llm.cache import get_cache_client


class LightRAGRetriever(BaseComponent):
    """
    Pure LightRAG retriever using LightRAG.query() directly.

    Uses LightRAG's native retrieval modes (naive, local, global, hybrid).
    Text-only, no multimodal processing.
    """

    name = "lightrag_retriever"
    _instances: ClassVar[Dict[str, Any]] = {}
    _init_locks: ClassVar[Dict[str, asyncio.Lock]] = {}

    def __init__(self, kb_base_dir: Optional[str] = None):
        """
        Initialize LightRAG retriever.

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
        """Get or create a pure LightRAG instance (text-only)."""
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
                from src.services.embedding import get_embedding_client
                from src.services.llm import get_llm_client

                from lightrag import LightRAG

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
                                "Failed to release LightRAG init lock: %s", exc
                            )

            except ImportError as e:
                self.logger.error(f"Failed to import LightRAG: {e}")
                raise

    async def process(
        self,
        query: str,
        kb_name: str,
        mode: str = "hybrid",
        only_need_context: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Search using pure LightRAG retrieval (text-only).

        Args:
            query: Search query
            kb_name: Knowledge base name
            mode: Search mode (hybrid, local, global, naive)
            only_need_context: Whether to only return context without answer
            **kwargs: Additional arguments

        Returns:
            Search results dictionary
        """
        self.logger.info(f"LightRAG search ({mode}) in {kb_name}: {query[:50]}...")

        from src.logging.adapters import LightRAGLogContext

        with LightRAGLogContext(scene="LightRAG-Search"):
            rag = await self._get_lightrag_instance(kb_name)

            # Initialize storages if not already initialized
            await rag.initialize_storages()
            from lightrag.kg.shared_storage import initialize_pipeline_status

            await initialize_pipeline_status()

            # Import QueryParam for proper query parameter passing
            from lightrag import QueryParam

            # Use LightRAG's native query method with QueryParam object
            query_param = QueryParam(mode=mode, only_need_context=only_need_context)
            answer = await rag.aquery(query, param=query_param)
            answer_str = answer if isinstance(answer, str) else str(answer)

            return {
                "query": query,
                "answer": answer_str,
                "content": answer_str,
                "mode": mode,
                "provider": "lightrag",
            }
