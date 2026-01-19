# -*- coding: utf-8 -*-
"""OpenAI-compatible embedding adapter for OpenAI, Azure, HuggingFace, LM Studio, etc."""

import logging
from typing import Any, cast
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

import httpx

from .base import BaseEmbeddingAdapter, EmbeddingRequest, EmbeddingResponse

logger = logging.getLogger(__name__)


class OpenAICompatibleEmbeddingAdapter(BaseEmbeddingAdapter):
    MODELS_INFO: dict[str, int | dict[str, int | list[int]]] = {
        "text-embedding-3-large": {
            "default": 3072,
            "dimensions": [256, 512, 1024, 3072],
        },
        "text-embedding-3-small": {"default": 1536, "dimensions": [512, 1536]},
        "text-embedding-ada-002": 1536,
    }

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        if not self.base_url:
            raise ValueError("Base URL is required for OpenAI-compatible embedding")

        headers = {
            "Content-Type": "application/json",
        }
        if self.api_version:
            if not self.api_key:
                raise ValueError("API key is required for Azure/OpenAI with api_version")
            headers["api-key"] = self.api_key
        elif self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload: dict[str, Any] = {
            "input": request.texts,
            "model": request.model or self.model,
            "encoding_format": request.encoding_format or "float",
        }

        if request.dimensions or self.dimensions:
            payload["dimensions"] = request.dimensions or self.dimensions

        # Handle URL construction with urllib.parse
        parsed_url = urlparse(self.base_url)
        path = parsed_url.path.rstrip("/")
        if not path.endswith("/embeddings"):
            path = f"{path}/embeddings"

        query_params = parse_qs(parsed_url.query)
        if self.api_version:
            query_params["api-version"] = [self.api_version]

        new_query = urlencode(query_params, doseq=True)
        url = urlunparse(
            (
                parsed_url.scheme,
                parsed_url.netloc,
                path,
                parsed_url.params,
                new_query,
                parsed_url.fragment,
            )
        )

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
        expected_dims = request.dimensions or self.dimensions

        if expected_dims and actual_dims != expected_dims:
            logger.warning(
                f"Dimension mismatch: expected {expected_dims}, got {actual_dims}. "
                f"Model '{data['model']}' may not support custom dimensions."
            )

        logger.info(
            f"Successfully generated {len(embeddings)} embeddings "
            f"(model: {data['model']}, dimensions: {actual_dims})"
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
                "provider": "openai_compatible",
            }

        if isinstance(model_info, dict):
            info_dict = cast(dict[str, Any], model_info)
            return {
                "model": model_name,
                "dimensions": info_dict.get("default", self.dimensions),
                "supported_dimensions": info_dict.get("dimensions", []),
                "supports_variable_dimensions": len(
                    cast(list[int], info_dict.get("dimensions", []))
                )
                > 1,
                "provider": "openai_compatible",
            }
        else:
            return {
                "model": model_name,
                "dimensions": model_info or self.dimensions,
                "supported_dimensions": [],
                "supports_variable_dimensions": False,
                "provider": "openai_compatible",
            }
