"""
Embedding Client
================

Unified embedding client for all DeepTutor services.
"""

from typing import List, Optional

from src.logging import get_logger
from .config import EmbeddingConfig, get_embedding_config


class EmbeddingClient:
    """
    Unified embedding client for all services.
    
    Wraps the underlying embedding API (OpenAI-compatible) with a consistent interface.
    """

    def __init__(self, config: Optional[EmbeddingConfig] = None):
        """
        Initialize embedding client.

        Args:
            config: Embedding configuration. If None, loads from environment.
        """
        self.config = config or get_embedding_config()
        self.logger = get_logger("EmbeddingClient")

    async def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings for texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        from lightrag.llm.openai import openai_embed

        return await openai_embed(
            texts,
            model=self.config.model,
            api_key=self.config.api_key,
            base_url=self.config.base_url,
        )

    def embed_sync(self, texts: List[str]) -> List[List[float]]:
        """
        Synchronous wrapper for embed().
        
        Use this when you need to call from non-async context.
        """
        import asyncio
        
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self.embed(texts))
                    return future.result()
            else:
                return loop.run_until_complete(self.embed(texts))
        except RuntimeError:
            return asyncio.run(self.embed(texts))

    def get_embedding_func(self):
        """
        Get an EmbeddingFunc compatible with LightRAG.
        
        Returns:
            EmbeddingFunc instance
        """
        from lightrag.utils import EmbeddingFunc
        from lightrag.llm.openai import openai_embed

        return EmbeddingFunc(
            embedding_dim=self.config.dim,
            max_token_size=self.config.max_tokens,
            func=lambda texts: openai_embed(
                texts,
                model=self.config.model,
                api_key=self.config.api_key,
                base_url=self.config.base_url,
            ),
        )


# Singleton instance
_client: Optional[EmbeddingClient] = None


def get_embedding_client(config: Optional[EmbeddingConfig] = None) -> EmbeddingClient:
    """
    Get or create the singleton embedding client.

    Args:
        config: Optional configuration. Only used on first call.

    Returns:
        EmbeddingClient instance
    """
    global _client
    if _client is None:
        _client = EmbeddingClient(config)
    return _client


def reset_embedding_client():
    """Reset the singleton embedding client."""
    global _client
    _client = None

