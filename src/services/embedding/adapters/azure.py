# -*- coding: utf-8 -*-
"""Azure OpenAI embedding adapter using the official SDK."""

import logging
from typing import Any

import openai

from .base import BaseEmbeddingAdapter, EmbeddingRequest, EmbeddingResponse

logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 100

MODELS_INFO: dict[str, dict[str, int | list[int]]] = {
    "text-embedding-3-large": {
        "default": 3072,
        "dimensions": [256, 512, 1024, 3072],
    },
    "text-embedding-3-small": {
        "default": 1536,
        "dimensions": [512, 1536],
    },
    "text-embedding-ada-002": {
        "default": 1536,
        "dimensions": [1536],
    },
}


def _chunk_texts(texts: list[str], batch_size: int) -> list[list[str]]:
    """
    Split texts into batches to respect provider limits.

    Args:
        texts: Input texts to embed.
        batch_size: Maximum number of texts per request.

    Returns:
        List of text batches.
    """
    if batch_size <= 0:
        return [texts]

    return [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]


def _merge_usage(total: dict[str, int], usage: dict[str, object]) -> None:
    """
    Merge usage counters into the total usage dictionary.

    Args:
        total: Mutable totals dictionary.
        usage: Usage payload from a single response.
    """
    for key, value in usage.items():
        if isinstance(value, int):
            total[key] = total.get(key, 0) + value


class AzureEmbeddingAdapter(BaseEmbeddingAdapter):
    """
    Adapter for Azure OpenAI Embedding API using the official SDK.
    Eliminates manual URL parsing and ensures consistent auth handling.

    Args:
        None.

    Returns:
        None.

    Raises:
        None.
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

        Raises:
            ValueError: If embedding dimensions mismatch expected dimensions.
            openai.APIError: If the Azure OpenAI SDK returns an API error.
        """
        self._validate_texts(request.texts)

        model = request.model or self.model
        if not model:
            raise ValueError("Model is required for Azure OpenAI embedding")

        embeddings: list[list[float]] = []
        usage_totals: dict[str, int] = {}
        response_model = model

        try:
            for batch in _chunk_texts(request.texts, DEFAULT_BATCH_SIZE):
                create_kwargs: dict[str, Any] = {
                    "input": batch,
                    "model": model,
                    "encoding_format": "float",
                }
                if request.dimensions is not None:
                    create_kwargs["dimensions"] = request.dimensions
                elif self.dimensions is not None:
                    create_kwargs["dimensions"] = self.dimensions

                response = await self.client.embeddings.create(**create_kwargs)

                embeddings.extend([item.embedding for item in response.data])
                response_model = response.model
                usage_payload = response.usage.model_dump() if response.usage else {}
                _merge_usage(usage_totals, usage_payload)

            if not embeddings:
                raise ValueError("Invalid API response: missing or empty 'data' field")

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
                    f"Embedding dimension mismatch: expected {expected_dims}, got {actual_dims}. "
                    "Check your model configuration and ensure the dimensions parameter "
                    "matches the model's output dimensions."
                )

            return EmbeddingResponse(
                embeddings=embeddings,
                model=response_model,
                dimensions=actual_dims,
                usage=usage_totals,
            )
        except openai.APIError as e:
            logger.error("Azure Embedding API Error: %s", e)
            raise
        except Exception as e:
            logger.exception("Unexpected error during Azure embedding: %s", e)
            raise

    def get_model_info(self) -> dict[str, Any]:
        """
        Return information about the configured model.

        Returns:
            Dictionary with model metadata

        Raises:
            None.
        """
        model_name = self.model or ""
        model_info = MODELS_INFO.get(model_name, {})
        supported_dimensions = model_info.get("dimensions", [])
        return {
            "model": model_name,
            "dimensions": model_info.get("default", self.dimensions),
            "provider": "azure",
            "client_wrapper": "openai-sdk",
            "supports_variable_dimensions": (
                len(supported_dimensions) > 1 if isinstance(supported_dimensions, list) else False
            ),
        }
