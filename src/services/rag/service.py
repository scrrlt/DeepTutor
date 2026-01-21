# -*- coding: utf-8 -*-
"""
RAG Service
===========

Unified RAG service providing a single entry point for all RAG operations.
"""

import asyncio
import json
import os
from pathlib import Path
import shutil
from typing import Any, Protocol

import aiofiles

from src.logging import get_logger

from .factory import get_pipeline, has_pipeline, list_pipelines

# Default knowledge base directory
DEFAULT_KB_BASE_DIR = str(
    Path(__file__).resolve().parent.parent.parent.parent / "data" / "knowledge_bases"
)


class RAGPipelineProtocol(Protocol):
    """
    Protocol for RAG pipeline behaviors used by the service.

    This avoids coupling to a concrete pipeline implementation while keeping
    typed access to initialize/search/delete calls.
    """

    async def initialize(self, kb_name: str, file_paths: list[str], **kwargs: Any) -> bool:
        """
        Initialize a knowledge base with the given documents.
        
        Parameters:
            kb_name (str): Name of the knowledge base to create or update.
            file_paths (list[str]): Paths to documents to ingest into the knowledge base.
            **kwargs: Additional provider-specific options forwarded to the underlying pipeline.
        
        Returns:
            true if initialization succeeded, false otherwise.
        """

    async def search(self, query: str, kb_name: str, **kwargs: Any) -> dict[str, Any]:
        """
        Perform a retrieval-augmented search against a knowledge base and return a normalized result payload.
        
        Parameters:
            query (str): The user's search query.
            kb_name (str): The name of the knowledge base to query.
            **kwargs: Additional provider-specific options (e.g., `mode`).
        
        Returns:
            dict[str, Any]: A normalized result dictionary that always includes the keys:
                - `query`: the original query string,
                - `answer`: the selected answer text (falls back to `content` if absent),
                - `content`: the context/content (falls back to `answer` if absent),
                - `provider`: the pipeline provider that served the request,
                - `mode`: the retrieval mode used.
        """

    async def delete(self, kb_name: str) -> bool:
        """
        Delete the named knowledge base from storage.
        
        Attempts to remove the knowledge base identified by `kb_name`; may use a provider-specific deletion implementation when available and otherwise removes the KB directory.
        
        Returns:
            bool: `True` if the knowledge base was deleted, `False` otherwise.
        """


class RAGService:
    """
    Unified RAG service entry point.

    Provides a clean interface for RAG operations:
    - Knowledge base initialization
    - Search/retrieval
    - Knowledge base deletion

    Usage:
        # Default configuration
        service = RAGService()
        await service.initialize("my_kb", ["doc1.pdf"])
        result = await service.search("query", "my_kb")

        # Custom configuration for testing
        service = RAGService(kb_base_dir="/tmp/test_kb", provider="llamaindex")
        await service.initialize("test", ["test.txt"])
    """

    def __init__(
        self,
        kb_base_dir: str | None = None,
        provider: str | None = None,
    ):
        """
        Create a RAGService configured with a knowledge-base base directory and a default pipeline provider.
        
        Parameters:
            kb_base_dir (str | None): Path to the base directory for knowledge bases. If None, uses DEFAULT_KB_BASE_DIR.
            provider (str | None): Default RAG pipeline provider name. If None, uses the RAG_PROVIDER environment variable or "raganything".
        """
        self.logger = get_logger("RAGService")
        self.kb_base_dir = kb_base_dir or DEFAULT_KB_BASE_DIR
        self.provider = provider or os.getenv("RAG_PROVIDER", "raganything")
        self._pipeline_cache: dict[str, RAGPipelineProtocol] = {}

    def _get_cached_pipeline(self, provider: str) -> RAGPipelineProtocol:
        """
        Retrieve a cached RAG pipeline instance for the given provider and the service's knowledge-base base directory.
        
        Parameters:
            provider (str): Name of the pipeline provider to retrieve.
        
        Returns:
            RAGPipelineProtocol: Pipeline instance associated with the specified provider and this service's `kb_base_dir`.
        """
        cache_key = f"{provider}:{self.kb_base_dir}"
        if cache_key not in self._pipeline_cache:
            self._pipeline_cache[cache_key] = get_pipeline(
                provider,
                kb_base_dir=self.kb_base_dir,
            )
        return self._pipeline_cache[cache_key]

    def _get_pipeline(self) -> RAGPipelineProtocol:
        """
        Retrieve the RAG pipeline instance for the service's current provider.
        
        @returns RAGPipelineProtocol: The pipeline instance associated with the service's configured provider and knowledge-base directory.
        """
        return self._get_cached_pipeline(self.provider)

    async def initialize(self, kb_name: str, file_paths: list[str], **kwargs: Any) -> bool:
        """
        Initialize a knowledge base by ingesting the provided documents into the configured pipeline.
        
        Parameters:
            kb_name (str): Name of the knowledge base to create or update.
            file_paths (list[str]): Paths to document files to ingest into the knowledge base.
            **kwargs: Additional provider-specific options forwarded to the pipeline's initialize method.
        
        Returns:
            bool: `True` if the knowledge base was initialized successfully, `False` otherwise.
        """
        self.logger.info(f"Initializing KB '{kb_name}' with provider '{self.provider}'")
        pipeline = self._get_pipeline()
        return await pipeline.initialize(kb_name=kb_name, file_paths=file_paths, **kwargs)

    async def search(
        self, query: str, kb_name: str, mode: str = "hybrid", **kwargs: Any
    ) -> dict[str, Any]:
        """
        Search a knowledge base and return a normalized result payload.
        
        Parameters:
            query (str): Search query text.
            kb_name (str): Knowledge base name to search.
            mode (str): Search mode to use; typically "hybrid", "local", "global", or "naive".
            **kwargs: Forwarded to the provider pipeline's search implementation.
        
        Returns:
            dict[str, Any]: Result dictionary containing at minimum the keys:
                - `query`: The original query.
                - `answer`: Generated answer (falls back to `content` if missing).
                - `content`: Retrieved content (falls back to `answer` if missing).
                - `mode`: The search mode used.
                - `provider`: The pipeline provider that served the query.
            May include additional provider-specific fields.
        """
        # Get the provider from KB metadata, fallback to instance provider
        provider = await self._get_provider_for_kb(kb_name)

        self.logger.info(
            f"Searching KB '{kb_name}' with provider '{provider}' and query: {query[:50]}..."
        )

        # Get pipeline for the specific provider
        pipeline = self._get_cached_pipeline(provider)

        result = await pipeline.search(query=query, kb_name=kb_name, mode=mode, **kwargs)

        # Ensure consistent return format
        if "query" not in result:
            result["query"] = query
        if "answer" not in result and "content" in result:
            result["answer"] = result["content"]
        if "content" not in result and "answer" in result:
            result["content"] = result["answer"]
        if "provider" not in result:
            result["provider"] = provider
        if "mode" not in result:
            result["mode"] = mode

        return result

    async def _get_provider_for_kb(self, kb_name: str) -> str:
        """
        Determine which RAG provider to use for a given knowledge base by inspecting its metadata.
        
        If the KB's metadata contains a `rag_provider` that corresponds to a known pipeline, that provider is returned; otherwise the service's configured provider is returned.
        
        Parameters:
            kb_name (str): Name of the knowledge base.
        
        Returns:
            str: Selected provider name (for example, 'llamaindex', 'lightrag', or 'raganything').
        """
        try:
            metadata_file = Path(self.kb_base_dir) / kb_name / "metadata.json"

            metadata_exists = metadata_file.exists()
            if metadata_exists:
                async with aiofiles.open(metadata_file, encoding="utf-8") as file_handle:
                    content = await file_handle.read()
                metadata = json.loads(content)
                provider = metadata.get("rag_provider")
                if provider:
                    if has_pipeline(provider):
                        self.logger.info(f"Using provider '{provider}' from KB metadata")
                        return provider
                    self.logger.warning(
                        f"Unknown provider '{provider}' in KB metadata; falling back to instance provider",
                    )

            # Fallback to instance provider
            self.logger.info(
                "No provider in metadata, using instance provider: %s",
                self.provider,
            )
            return self.provider

        except Exception as e:
            self.logger.warning(
                f"Error reading provider from metadata: {e}, using instance provider"
            )
            return self.provider

    async def delete(self, kb_name: str) -> bool:
        """
        Delete the named knowledge base.
        
        Attempts to use the configured pipeline's `delete` method; if not available, removes the KB directory from the service's base directory.
        
        Parameters:
            kb_name (str): Name of the knowledge base to delete.
        
        Returns:
            bool: `True` if the knowledge base was deleted, `False` otherwise.
        """
        self.logger.info(f"Deleting KB '{kb_name}'")
        pipeline = self._get_pipeline()

        if hasattr(pipeline, "delete"):
            return await pipeline.delete(kb_name=kb_name)

        # Fallback: delete directory manually
        kb_dir = Path(self.kb_base_dir) / kb_name
        kb_exists = kb_dir.exists()
        if kb_exists:
            await asyncio.to_thread(shutil.rmtree, kb_dir)
            self.logger.info(f"Deleted KB directory: {kb_dir}")
            return True
        return False

    @staticmethod
    def list_providers() -> list[dict[str, str]]:
        """
        Return a list of available RAG pipeline providers.
        
        Returns:
            list[dict[str, str]]: A list of provider info dictionaries, each containing at least
                "id" (provider identifier) and "description" (human-readable description).
        """
        return list_pipelines()

    @staticmethod
    def get_current_provider() -> str:
        """
        Get the currently configured default provider.

        Returns:
            Provider name from RAG_PROVIDER env var or default
        """
        return os.getenv("RAG_PROVIDER", "raganything")

    @staticmethod
    def has_provider(name: str) -> bool:
        """
        Check if a provider is available.

        Args:
            name: Provider name

        Returns:
            True if provider exists
        """
        return has_pipeline(name)