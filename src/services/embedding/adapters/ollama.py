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
    Scale a numeric vector to unit Euclidean length.
    
    If the vector's Euclidean norm is zero, returns the original vector unchanged.
    
    Parameters:
        vector (list[float]): Input embedding vector.
    
    Returns:
        list[float]: The vector normalized to length 1, or the original vector if its norm is zero.
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
    Return embeddings normalized to unit length when requested.
    
    Parameters:
        embeddings (list[list[float]]): List of embedding vectors.
        normalize (bool | None): If True, scale each vector to unit length; if False or None, return embeddings unchanged.
    
    Returns:
        list[list[float]]: The normalized embeddings when `normalize` is True, otherwise the original embeddings.
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
        Generate embeddings for the provided texts using the local Ollama API.
        
        Parameters:
            request (EmbeddingRequest): Request containing input texts and optional parameters
                such as model, dimensions, truncate, and normalized.
        
        Returns:
            EmbeddingResponse: Response containing generated embeddings, the model name,
            the embedding dimensionality, and usage metadata (`prompt_eval_count`, `total_duration`).
        
        Raises:
            ValueError: If base_url is not configured or the requested model is not found.
            ConnectionError: If unable to connect to the Ollama server.
            TimeoutError: If the request to Ollama times out.
            httpx.HTTPError: For other HTTP-related errors from the Ollama API.
        """
        if not self.base_url:
            raise ValueError("Base URL is required for Ollama embedding")

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
                            raise ValueError(
                                f"Model '{payload['model']}' not found in Ollama. "
                                f"Available models: {', '.join(available_models[:10])}. "
                                f"Download it with: ollama pull {payload['model']}"
                            )
                    except httpx.HTTPError:
                        pass

                    raise ValueError(
                        f"Model '{payload['model']}' not found. "
                        f"Download it with: ollama pull {payload['model']}"
                    )

                response.raise_for_status()
                data = response.json()

        except httpx.ConnectError as e:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.base_url}. "
                f"Make sure Ollama is running. Start it with: ollama serve"
            ) from e

        except httpx.TimeoutException as e:
            raise TimeoutError(
                f"Request to Ollama timed out after {self.request_timeout}s. "
                f"The model might be too large or the server is overloaded."
            ) from e

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
            f"Successfully generated {len(embeddings)} embeddings "
            f"(model: {data.get('model', self.model)}, dimensions: {actual_dims})"
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
        Provide metadata about the adapter's configured model.
        
        Returns:
            dict: Metadata dictionary with the following keys:
                - model (str): Configured model name or empty string if unset.
                - dimensions (int): Expected embedding dimensionality (from MODELS_INFO or adapter fallback).
                - supported_dimensions (list): List of supported dimensions (empty for local Ollama).
                - local (bool): True indicating the model runs locally.
                - supports_variable_dimensions (bool): False for Ollama (no variable-dimension support).
                - provider (str): Provider identifier, always "ollama".
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