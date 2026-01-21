# -*- coding: utf-8 -*-
"""Ollama Embedding Adapter for local embeddings."""

import logging
import math
from typing import Any, cast

import httpx

from .base import BaseEmbeddingAdapter, EmbeddingRequest, EmbeddingResponse

logger = logging.getLogger(__name__)


def _normalize_vector(vector: list[float]) -> list[float]:
    """
    Normalize a vector to unit length.

    Args:
        vector: Embedding vector.

    Returns:
        Normalized vector (or original if norm is zero).
    """
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def _normalize_embeddings(
    embeddings: list[list[float]],
    normalize: bool | None,
) -> list[list[float]]:
    """
    Normalize embeddings when requested.

    Args:
        embeddings: Raw embedding vectors.
        normalize: Whether to normalize embeddings.

    Returns:
        Normalized embeddings when enabled.
    """
    if normalize is True:
        return [_normalize_vector(vector) for vector in embeddings]

    return embeddings


class OllamaEmbeddingAdapter(BaseEmbeddingAdapter):
    """Adapter for the Ollama embedding API."""

    MODELS_INFO: dict[str, int] = {
        "all-minilm": 384,
        "all-mpnet-base-v2": 768,
        "nomic-embed-text": 768,
        "mxbai-embed-large": 1024,
        "snowflake-arctic-embed": 1024,
    }

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """
        Generate embeddings using Ollama local API.

        Args:
            request: EmbeddingRequest containing texts and parameters

        Returns:
            EmbeddingResponse with embeddings and metadata

        Raises:
            ValueError: If required configuration or response data is missing.
            httpx.HTTPError: If the Ollama API request fails.
        """
        # Validate texts input
        self._validate_texts(request.texts)

        if not self.base_url:
            raise ValueError("Base URL is required for Ollama embedding")  # noqa: TRY003

        payload: dict[str, Any] = {
            "model": request.model or self.model,
            "input": request.texts,
        }

        if request.dimensions or self.dimensions:
            payload["dimensions"] = request.dimensions or self.dimensions

        if request.truncate is not None:
            payload["truncate"] = request.truncate

        payload["keep_alive"] = "5m"

        url = f"{self.base_url.rstrip('/')}/api/embed"

        logger.debug(
            "Sending embedding request to %s with %d texts",
            url,
            len(request.texts),
        )

        try:
            async with httpx.AsyncClient(timeout=self.request_timeout) as client:
                response = await client.post(url, json=payload)

                if response.status_code == 404:
                    try:
                        health_check = await client.get(f"{self.base_url.rstrip('/')}/api/tags")
                        if health_check.status_code == 200:
                            available_models = [
                                cast(str, m.get("name", ""))
                                for m in cast(
                                    list[dict[str, Any]],
                                    health_check.json().get("models", []),
                                )
                            ]
                            logger.debug(
                                "Model '%s' not found in Ollama. Available models: %s",
                                payload["model"],
                                ", ".join(available_models[:10])[:200],
                            )
                            raise ValueError("Model not found")  # noqa: TRY003
                    except httpx.HTTPError:
                        pass

                    logger.debug(
                        "Model '%s' not found. Ask user to download it with: ollama pull %s",
                        payload["model"],
                        payload["model"],
                    )
                    raise ValueError("Model not found")  # noqa: TRY003

                response.raise_for_status()
                data = response.json()

        except httpx.ConnectError as e:
            logger.error("Cannot connect to Ollama at %s", self.base_url)
            raise ConnectionError("Cannot connect to Ollama") from e  # noqa: TRY003

        except httpx.TimeoutException as e:
            logger.exception("Ollama request timed out after %ss", self.request_timeout)
            raise TimeoutError("Ollama request timed out") from e  # noqa: TRY003

        except httpx.HTTPError as e:
            logger.error("Ollama API error: %s", e)
            raise

        embeddings = _normalize_embeddings(
            data["embeddings"],
            request.normalized,
        )

        actual_dims = len(embeddings[0]) if embeddings else 0
        expected_dims = request.dimensions or self.dimensions

        if expected_dims and actual_dims != expected_dims:
            logger.warning(
                f"Dimension mismatch: expected {expected_dims}, got {actual_dims}. "
                f"Model '{payload['model']}' may not support custom dimensions."
            )

        logger.info(
            "Successfully generated %d embeddings " "(model: %s, dimensions: %d)",
            len(embeddings),
            data.get("model", self.model),
            actual_dims,
        )

        return EmbeddingResponse(
            embeddings=embeddings,
            model=data.get("model", self.model or "unknown"),
            dimensions=actual_dims,
            usage={
                "prompt_eval_count": data.get("prompt_eval_count", 0),
                "total_duration": data.get("total_duration", 0),
            },
        )

    def get_model_info(self) -> dict[str, Any]:
        """
        Return information about the configured model.

        Returns:
            Dictionary with model metadata (name, dimensions, etc.)
        """
        model_name = self.model or ""
        return {
            "model": model_name,
            "dimensions": self.MODELS_INFO.get(model_name, self.dimensions or 0),
            "supported_dimensions": [],
            "local": True,
            "supports_variable_dimensions": False,
            "provider": "ollama",
        }
