# -*- coding: utf-8 -*-
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
        """
        Initialize the Azure OpenAI embedding adapter and construct the AsyncAzureOpenAI client.
        
        Expects the provided configuration to supply authentication and endpoint settings. Requires an API key and a base URL; optionally accepts an API version and request timeout. Constructs self.client as an AsyncAzureOpenAI instance using the adapter's resolved api_key, base_url (as azure_endpoint), api_version (defaulting to "2023-05-15" if not set), and request_timeout.
        
        Parameters:
            config (dict[str, Any]): Adapter configuration used by the base class to populate attributes like `api_key`, `base_url`, `api_version`, and `request_timeout`.
        
        Raises:
            ValueError: If the API key or base URL is missing from the configuration.
        """
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
        Generate embeddings for the provided texts using the configured Azure OpenAI model.
        
        Parameters:
            request (EmbeddingRequest): Input texts and optional per-request settings (model, dimensions, encoding_format).
        
        Returns:
            EmbeddingResponse: Contains the list of embeddings, the model name, actual embedding dimensions, and usage metadata (empty dict if not provided).
        
        Raises:
            ValueError: If `request.texts` is empty, if the API response has no usable data, or if the returned embedding dimensionality does not match the expected dimensions.
            openai.APIError: If the underlying Azure OpenAI SDK reports an API error (re-raised).
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
        Provide metadata about the adapter's currently configured embedding model.
        
        Returns:
            info (dict[str, Any]): Dictionary with keys:
                - "model": configured model name (str)
                - "dimensions": expected embedding dimensionality (int | None)
                - "provider": provider identifier, "azure" (str)
                - "client_wrapper": client wrapper identifier, "openai-sdk" (str)
                - "supports_variable_dimensions": whether the adapter supports variable-length embeddings (`True`)
        """
        return {
            "model": self.model,
            "dimensions": self.dimensions,
            "provider": "azure",
            "client_wrapper": "openai-sdk",
            "supports_variable_dimensions": True,
        }