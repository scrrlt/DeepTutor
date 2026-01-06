"""
Embedding Service
=================

Unified embedding client for all DeepTutor modules.

Usage:
    from src.services.embedding import get_embedding_client, EmbeddingClient, EmbeddingConfig

    # Get singleton client
    client = get_embedding_client()
    vectors = await client.embed(["text1", "text2"])

    # Get LightRAG-compatible EmbeddingFunc
    embed_func = client.get_embedding_func()
"""

from .client import EmbeddingClient, get_embedding_client, reset_embedding_client
from .config import EmbeddingConfig, get_embedding_config

__all__ = [
    "EmbeddingClient",
    "EmbeddingConfig",
    "get_embedding_client",
    "get_embedding_config",
    "reset_embedding_client",
]

