# -*- coding: utf-8 -*-
"""Jina AI embedding adapter with task-aware embeddings and late chunking."""

import logging
from typing import Any, cast

import httpx

from .base import BaseEmbeddingAdapter, EmbeddingRequest, EmbeddingResponse

logger = logging.getLogger(__name__)


class JinaEmbeddingAdapter(BaseEmbeddingAdapter):
    """
    Adapter for the Jina embedding API.

    Args:
        None.

    Returns:
        None.

    """

    MODELS_INFO: dict[str, dict[str, Any]] = {
        "jina-embeddings-v3": {
            "default": 1024,
            "dimensions": [32, 64, 128, 256, 512, 768, 1024],
        },
        "jina-embeddings-v4": {
            "default": 1024,
            "dimensions": [32, 64, 128, 256, 512, 768, 1024],
        },
    }

    INPUT_TYPE_TO_TASK = {
        "search_document": "retrieval.passage",
        "search_query": "retrieval.query",
        "classification": "classification",
        "clustering": "separation",
        "text-matching": "text-matching",
    }

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """
        Generate embeddings using Jina AI API.

        Args:
            request: EmbeddingRequest containing texts and parameters

        Returns:
            EmbeddingResponse with embeddings and metadata

        Raises:
            ValueError: If required configuration or response data is missing.
            httpx.HTTPError: If the Jina API request fails.
        """
        if not self.api_key:
            raise ValueError("API key is required for Jina embedding")
        if not self.base_url:
            raise ValueError("Base URL is required for Jina embedding")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload: dict[str, Any] = {
            "input": request.texts,
            "model": request.model or self.model,
        }

        if request.dimensions:
            payload["dimensions"] = request.dimensions
        elif self.dimensions:
            payload["dimensions"] = self.dimensions

        if request.input_type:
            task = self.INPUT_TYPE_TO_TASK.get(request.input_type, request.input_type)
            payload["task"] = task
            logger.debug("Using Jina task: %s", task)

        if request.normalized is not None:
            payload["normalized"] = request.normalized

        if request.late_chunking:
            payload["late_chunking"] = True

        url = f"{self.base_url}/embeddings"

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

        if "data" not in data or not data["data"]:
            raise ValueError("Invalid API response: missing or empty 'data' field")

        embeddings = [item["embedding"] for item in data["data"]]
        actual_dims = len(embeddings[0]) if embeddings else 0

        logger.info(
            "Successfully generated %d embeddings (model: %s, dimensions: %d)",
            len(embeddings),
            data["model"],
            actual_dims,
        )

        return EmbeddingResponse(
            embeddings=embeddings,
            model=data["model"],
            dimensions=actual_dims,
            usage=data.get("usage", {}),
        )

    def get_model_info(self) -> dict[str, Any]:
        """
        Return information about the configured model.

        Returns:
            Dictionary with model metadata (name, dimensions, etc.)
        """
        model_name = self.model or ""
        model_info = self.MODELS_INFO.get(model_name, self.dimensions)

        if model_info is None and not model_name:
            return {
                "model": "unknown",
                "dimensions": None,
                "supported_dimensions": [],
                "supports_variable_dimensions": False,
                "provider": "jina",
            }

        if isinstance(model_info, dict):
            info_dict = cast(dict[str, Any], model_info)
            return {
                "model": model_name,
                "dimensions": info_dict.get("default", self.dimensions),
                "supported_dimensions": info_dict.get("dimensions", []),
                "supports_variable_dimensions": True,
                "provider": "jina",
            }
        else:
            return {
                "model": model_name,
                "dimensions": model_info or self.dimensions,
                "supported_dimensions": [],
                "supports_variable_dimensions": False,
                "provider": "jina",
            }
