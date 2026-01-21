# -*- coding: utf-8 -*-
"""Cohere Embedding Adapter for v1 and v2 API."""

import logging
import math
from typing import Any, cast

import httpx

from .base import BaseEmbeddingAdapter, EmbeddingRequest, EmbeddingResponse

logger = logging.getLogger(__name__)


def _normalize_vector(vector: list[float]) -> list[float]:
    """
    Normalize a numeric vector to unit length.
    
    Parameters:
        vector (list[float]): Input embedding vector.
    
    Returns:
        list[float]: Vector scaled to have Euclidean norm 1, or the original vector if its norm is zero.
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
    Optionally normalize a list of embedding vectors to unit length.
    
    Parameters:
        embeddings (list[list[float]]): Embedding vectors to process.
        normalize (bool | None): If False, return `embeddings` unchanged; otherwise scale each vector to unit length. Vectors with zero Euclidean norm are returned unchanged.
    
    Returns:
        list[list[float]]: The processed list of embedding vectors (normalized when enabled).
    """
    if normalize is False:
        return embeddings
    return [_normalize_vector(vector) for vector in embeddings]


class CohereEmbeddingAdapter(BaseEmbeddingAdapter):
    """
    Adapter for Cohere Embed API (v1 and v2).

    Args:
        None.

    Returns:
        None.
    """

    MODELS_INFO: dict[str, dict[str, Any]] = {
        "embed-v4.0": {
            "dimensions": [256, 512, 1024, 1536],
            "default": 1024,
            "api_version": "v2",
        },
        "embed-english-v3.0": {
            "dimensions": [1024],
            "default": 1024,
            "api_version": "v1",
        },
        "embed-multilingual-v3.0": {
            "dimensions": [1024],
            "default": 1024,
            "api_version": "v1",
        },
        "embed-multilingual-light-v3.0": {
            "dimensions": [384],
            "default": 384,
            "api_version": "v1",
        },
        "embed-english-light-v3.0": {
            "dimensions": [384],
            "default": 384,
            "api_version": "v1",
        },
    }

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """
        Generate embeddings for provided texts using Cohere's embed API.
        
        Parameters:
            request (EmbeddingRequest): Request containing texts and optional parameters
                such as model, dimensions, input_type, truncate, and normalized.
        
        Returns:
            EmbeddingResponse: Contains the list of embeddings, the model name (from the
            response or configured fallback), the actual embedding dimensionality, and
            any billing/usage metadata from the API response.
        
        Raises:
            ValueError: If API key, base URL, or model name is missing, or if the API
                response does not include a non-empty `embeddings` field.
            httpx.HTTPError: If the HTTP request to the Cohere API fails.
        """
        if not self.api_key:
            raise ValueError("API key is required for Cohere embedding")
        if not self.base_url:
            raise ValueError("Base URL is required for Cohere embedding")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        model_name = request.model or self.model
        if not model_name:
            raise ValueError("Model name is required for Cohere embedding")
        model_info = self.MODELS_INFO.get(model_name, {})
        api_version = model_info.get("api_version", "v2")
        dimension = request.dimensions or self.dimensions

        input_type = request.input_type or "search_document"

        payload: dict[str, Any]
        if api_version == "v1":
            payload = {
                "texts": request.texts,
                "model": model_name,
                "input_type": input_type,
            }

            if not request.truncate:
                payload["truncate"] = "NONE"
        else:
            payload = {
                "texts": request.texts,
                "model": model_name,
                "embedding_types": ["float"],
                "input_type": input_type,
            }

            supported_dims = model_info.get("dimensions", [])
            if isinstance(supported_dims, list) and len(supported_dims) > 1:
                payload["output_dimension"] = dimension or model_info.get("default")

            if not request.truncate:
                payload["truncate"] = "NONE"

        url = f"{self.base_url.rstrip('/')}/{api_version}/embed"

        logger.debug(
            "Sending embedding request to %s with %d texts",
            url,
            len(request.texts),
        )

        async with httpx.AsyncClient(timeout=self.request_timeout) as client:
            response = await client.post(url, json=payload, headers=headers)

            if response.status_code >= 400:
                logger.error(
                    "HTTP %d response body: %s",
                    response.status_code,
                    response.text,
                )

            response.raise_for_status()
            data = response.json()

        if "embeddings" not in data or not data["embeddings"]:
            raise ValueError("Invalid API response: missing or empty 'embeddings' field")

        if api_version == "v1":
            embeddings = cast(list[list[float]], data["embeddings"])
        else:
            embeddings = cast(list[list[float]], data["embeddings"]["float"])

        embeddings = _normalize_embeddings(embeddings, request.normalized)

        actual_dims = len(embeddings[0]) if embeddings else 0
        expected_dims = request.dimensions or self.dimensions

        if expected_dims and actual_dims != expected_dims:
            logger.warning(
                "Dimension mismatch: expected %d, got %d",
                expected_dims,
                actual_dims,
            )

        model_from_data = data.get("model") or self.model or "unknown"
        logger.info(
            "Successfully generated %d embeddings (model: %s, dimensions: %d)",
            len(embeddings),
            model_from_data,
            actual_dims,
        )

        return EmbeddingResponse(
            embeddings=embeddings,
            model=model_from_data,
            dimensions=actual_dims,
            usage=data.get("meta", {}).get("billed_units", {}),
        )

    def get_model_info(self) -> dict[str, Any]:
        """
        Provide metadata for the adapter's configured model.
        
        Returns:
            info (dict[str, Any]): Dictionary containing:
                - model: Configured model name or empty string.
                - dimensions: Default or configured output dimension, or None when unknown.
                - supported_dimensions: List of supported output dimensions (may be empty).
                - supports_variable_dimensions: `True` when multiple supported dimensions are available, `False` otherwise.
                - provider: The string "cohere".
            
            When no model name and no dimensions are configured, `dimensions` is `None`,
            `supported_dimensions` is an empty list, and `model` is "unknown".
        """
        model_name = self.model or ""
        model_info = self.MODELS_INFO.get(model_name, {})

        if not model_info and not model_name and not self.dimensions:
            return {
                "model": "unknown",
                "dimensions": None,
                "supported_dimensions": [],
                "supports_variable_dimensions": False,
                "provider": "cohere",
            }

        dimensions_list = model_info.get("dimensions", [])
        return {
            "model": model_name,
            "dimensions": model_info.get("default", self.dimensions),
            "supported_dimensions": dimensions_list,
            "supports_variable_dimensions": (
                len(dimensions_list) > 1 if isinstance(dimensions_list, list) else False
            ),
            "provider": "cohere",
        }