"""Azure OpenAI embedding adapter using the official SDK."""

import logging
from typing import Any

import openai

from .base import BaseEmbeddingAdapter, EmbeddingRequest, EmbeddingResponse

logger = logging.getLogger(__name__)


class AzureEmbeddingAdapter(BaseEmbeddingAdapter):
    """
    Adapter for Azure OpenAI Embedding API using the official SDK.
    Eliminates manual URL parsing and ensures consistent auth handling.
    """

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)

        if not self.api_key:
            raise ValueError("API key is required for Azure OpenAI embedding")
        if not self.base_url:
            raise ValueError("Base URL is required for Azure OpenAI embedding")

        self.client = openai.AsyncAzureOpenAI(
            api_key=self.api_key,
            azure_endpoint=self.base_url,
            api_version=self.api_version or "2023-05-15",
            timeout=self.request_timeout,
        )

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """
        Generate embeddings using Azure OpenAI SDK.

        Args:
            request: EmbeddingRequest containing texts and parameters

        Returns:
            EmbeddingResponse with embeddings and metadata
        """
        # Validate request early
        if not request.texts:
            raise ValueError("Embedding request requires at least one text")

        try:
            # SDK handles auth headers, query params, and retries automatically
            response = await self.client.embeddings.create(
                input=request.texts,
                model=self.model,
                dimensions=request.dimensions or self.dimensions,
                encoding_format=request.encoding_format or "float",
            )

            # Standardize response
            data = response.data
            if not isinstance(data, (list, tuple)) or not data:
                logger.error("Azure Embedding API returned empty or invalid 'data' field")
                raise ValueError("Invalid API response: missing or empty 'data' field")

            embeddings = [item.embedding for item in data]

            # Hard validation for critical dimension drift
            actual_dims = len(embeddings[0]) if embeddings else 0
            expected_dims = request.dimensions or self.dimensions

            if expected_dims and actual_dims != expected_dims:
                logger.error(
                    "CRITICAL: Dimension mismatch. Expected %d, got %d. "
                    "This will corrupt the vector index.",
                    expected_dims,
                    actual_dims,
                )
                raise ValueError(
                    f"Embedding dimension mismatch: expected {expected_dims}, got {actual_dims}"
                )

            return EmbeddingResponse(
                embeddings=embeddings,
                model=response.model,
                dimensions=actual_dims,
                usage=response.usage.model_dump() if response.usage else {},
            )

        except openai.APIError as e:
            logger.error("Azure Embedding API Error: %s", e)
            raise

    def get_model_info(self) -> dict[str, Any]:
        """
        Return information about the configured model.

        Returns:
            Dictionary with model metadata
        """
        return {
            "model": self.model,
            "dimensions": self.dimensions,
            "provider": "azure",
            "client_wrapper": "openai-sdk",
            "supports_variable_dimensions": True,
        }
