#!/usr/bin/env python
"""
RAG Query Tool - Pure tool wrapper for RAG operations

This module provides simple function wrappers for RAG operations.
All logic is delegated to RAGService in src/services/rag/service.py.
"""

import asyncio
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Load environment variables
project_root = Path(__file__).parent.parent.parent
load_dotenv(project_root / "DeepTutor.env", override=False)
load_dotenv(project_root / ".env", override=False)

# Import RAGService as the single entry point
from src.logging import get_logger
from src.services.rag.service import RAGService

# Default provider constant used by tests and external callers
DEFAULT_RAG_PROVIDER = os.getenv("RAG_PROVIDER", "raganything")


async def rag_search(
    query: str,
    kb_name: str | None = None,
    mode: str = "hybrid",
    provider: str | None = None,
    kb_base_dir: str | None = None,
    **kwargs,
) -> dict:
    """
    Query knowledge base using configurable RAG pipeline.

    Args:
        query: Query question
        kb_name: Knowledge base name (optional, defaults to default knowledge base)
        mode: Query mode (e.g., "hybrid", "local", "global", "naive")
        provider: RAG pipeline to use (defaults to RAG_PROVIDER env var or "raganything")
        kb_base_dir: Base directory for knowledge bases (for testing)
        **kwargs: Additional parameters passed to the RAG pipeline

    Returns:
        dict: Dictionary containing query results
            {
                "query": str,
                "answer": str,
                "content": str,
                "mode": str,
                "provider": str
            }

    Raises:
        ValueError: If the specified RAG pipeline is not found
        Exception: If the query fails

    Example:
        # Use default provider (from .env)
        result = await rag_search("What is machine learning?", kb_name="textbook")

        # Override provider
        result = await rag_search("What is ML?", kb_name="textbook", provider="lightrag")
    """
    service = RAGService(kb_base_dir=kb_base_dir, provider=provider)

    try:
        return await service.search(query=query, kb_name=kb_name, mode=mode, **kwargs)
    except ValueError:
        # Preserve ValueError for callers/tests that expect the specific error type
        raise
    except Exception as e:
        raise Exception(f"RAG search failed: {e}")


async def initialize_rag(
    kb_name: str,
    documents: list[str],
    provider: str | None = None,
    kb_base_dir: str | None = None,
    **kwargs,
) -> bool:
    """
    Initialize RAG with documents.

    Args:
        kb_name: Knowledge base name
        documents: List of document file paths to index
        provider: RAG pipeline to use (defaults to RAG_PROVIDER env var)
        kb_base_dir: Base directory for knowledge bases (for testing)
        **kwargs: Additional arguments passed to pipeline

    Returns:
        True if successful

    Example:
        documents = ["doc1.pdf", "doc2.txt"]
        success = await initialize_rag("my_kb", documents)
    """
    service = RAGService(kb_base_dir=kb_base_dir, provider=provider)
    return await service.initialize(kb_name=kb_name, file_paths=documents, **kwargs)


async def delete_rag(
    kb_name: str,
    provider: str | None = None,
    kb_base_dir: str | None = None,
) -> bool:
    """
    Delete a knowledge base.

    Args:
        kb_name: Knowledge base name
        provider: RAG pipeline to use (defaults to RAG_PROVIDER env var)
        kb_base_dir: Base directory for knowledge bases (for testing)

    Returns:
        True if successful

    Example:
        success = await delete_rag("old_kb")
    """
    service = RAGService(kb_base_dir=kb_base_dir, provider=provider)
    return await service.delete(kb_name=kb_name)


def get_available_providers() -> list[dict[str, Any]]:
    """
    Get list of available RAG pipelines.

    Returns:
        List of pipeline information dictionaries

    Example:
        providers = get_available_providers()
        for p in providers:
            logger.info(f"{p['name']}: {p['description']}")
    """
    return RAGService.list_providers()


def get_current_provider() -> str:
    """Get the currently configured RAG provider"""
    return RAGService.get_current_provider()


# Backward compatibility aliases
get_available_plugins = get_available_providers
list_providers = RAGService.list_providers


if __name__ == "__main__":
    import sys

    if sys.platform == "win32":
        import io

        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    # List available providers
    logger.info("Available RAG Pipelines:")
    for provider in get_available_providers():
        logger.info("  - %s: %s", provider["id"], provider["description"])
    logger.info("Current provider: %s", get_current_provider())

    # Test search (requires existing knowledge base)
    result = asyncio.run(
        rag_search(
            "What is the lookup table (LUT) in FPGA?",
            kb_name="DE-all",
            mode="naive",
        )
    )

    logger.info("Query: %s", result["query"])
    logger.info("Answer: %s", result["answer"])
    logger.info("Provider: %s", result.get("provider", "unknown"))
