# -*- coding: utf-8 -*-
"""
Embedding Client
================

Unified embedding client for all DeepTutor services.
Now supports multiple providers through adapters.
"""

import asyncio
from typing import List, Optional

from src.logging import get_logger

from .adapters.base import EmbeddingRequest
from .config import EmbeddingConfig, get_embedding_config
from .provider import EmbeddingProviderManager, get_embedding_provider_manager


class EmbeddingClient:
    """
    Unified embedding client for all services.

    Delegates to provider-specific adapters based on configuration.
    Supports: OpenAI, Azure OpenAI, Cohere, Ollama, Jina, HuggingFace, Google.
    """

    def __init__(self, config: Optional[EmbeddingConfig] = None):
        """
        Create and configure an EmbeddingClient, loading adapters and capturing the initialization event loop.
        
        Parameters:
            config (Optional[EmbeddingConfig]): Configuration for the client. If omitted, the default configuration is loaded.
        
        Raises:
            Exception: If adapter initialization fails.
        """
        self.config = config or get_embedding_config()
        self.logger = get_logger("EmbeddingClient")
        self.manager: EmbeddingProviderManager = get_embedding_provider_manager()

        # Capture the loop where the client/adapters were created for thread-safe sync wrapper
        try:
            self._init_loop = asyncio.get_running_loop()
        except RuntimeError:
            self._init_loop = None

        # Initialize adapter based on binding configuration
        try:
            adapter = self.manager.get_adapter(
                self.config.binding,
                {
                    "api_key": self.config.api_key,
                    "base_url": self.config.base_url,
                    "api_version": getattr(self.config, "api_version", None),
                    "model": self.config.model,
                    "dimensions": self.config.dim,
                    "request_timeout": self.config.request_timeout,
                },
            )
            self.manager.set_adapter(adapter)

            self.logger.info(
                f"Initialized embedding client with {self.config.binding} adapter "
                f"(model: {self.config.model}, dimensions: {self.config.dim})"
            )
        except Exception as e:
            self.logger.error(f"Failed to initialize embedding adapter: {e}")
            raise

    async def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings for texts using the configured adapter.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        adapter = self.manager.get_active_adapter()

        request = EmbeddingRequest(
            texts=texts,
            model=self.config.model,
            dimensions=self.config.dim,
            input_type=self.config.input_type,  # Pass input_type for task-aware embeddings
        )

        try:
            response = await adapter.embed(request)

            self.logger.debug(
                f"Generated {len(response.embeddings)} embeddings using {self.config.binding}"
            )

            return response.embeddings
        except Exception as e:
            self.logger.error(f"Embedding request failed: {e}")
            raise

    def embed_sync(self, texts: List[str]) -> List[List[float]]:
        """
        Synchronously obtain embeddings, honoring the event loop where the client was initialized.
        
        If there is no running event loop in the current thread, the call runs the embedding request on a fresh loop and returns the results. If the client was initialized with a running loop and that loop is still active, the request is dispatched to the initialization loop and this call waits for the result (up to `self.config.request_timeout` or 30 seconds). If called from the same running loop that initialized the client, the call raises to avoid blocking the event loop. If a different loop is running but the client's initialization loop is no longer running, the call raises to indicate the client must be re-initialized in the current async context.
        
        Returns:
            List[List[float]]: A list of embedding vectors corresponding to the provided `texts`.
        
        Raises:
            RuntimeError: If called from the same running event loop that initialized the client.
            RuntimeError: If called from a running event loop while the client's initialization loop is no longer running.
        """
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None

        if current_loop is None:
            # No running loop, safe to create a new one
            return asyncio.run(self.embed(texts))

        if current_loop == self._init_loop:
            # Called sync from within the init loop - this would block
            raise RuntimeError(
                "embed_sync() cannot be called from within the same running event loop. "
                "Use 'await embed()' instead."
            )

        if self._init_loop and self._init_loop.is_running():
            # Different loop context, dispatch to init loop for adapter affinity
            future = asyncio.run_coroutine_threadsafe(self.embed(texts), self._init_loop)
            return future.result(timeout=self.config.request_timeout or 30)

        # Init loop is dead but we're in a different running loop - can't safely proceed
        raise RuntimeError(
            "embed_sync() called from a running event loop, but the initialization loop "
            "is no longer running. Re-initialize the client in the current async context."
        )

    def get_embedding_func(self):
        """
        Return an EmbeddingFunc compatible with LightRAG that produces NumPy arrays of embeddings.
        
        Returns:
            EmbeddingFunc: a configured embedding function that returns embeddings as a NumPy array and uses this client's embedding dimension and max token size.
        """
        from lightrag.utils import EmbeddingFunc
        import numpy as np

        # Create async wrapper that uses our adapter system
        # LightRAG expects numpy arrays, not Python lists
        async def embedding_wrapper(texts: List[str]):
            embeddings = await self.embed(texts)
            # Convert list of lists to numpy array for LightRAG compatibility
            return np.array(embeddings)

        return EmbeddingFunc(
            embedding_dim=self.config.dim,
            max_token_size=self.config.max_tokens,
            func=embedding_wrapper,
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