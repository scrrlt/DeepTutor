# -*- coding: utf-8 -*-
"""Azure OpenAI embedding adapter."""

import logging
from typing import Any, Dict, cast
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

import httpx

from .base import BaseEmbeddingAdapter, EmbeddingRequest, EmbeddingResponse

logger = logging.getLogger(__name__)


class AzureEmbeddingAdapter(BaseEmbeddingAdapter):
    """
    Adapter for Azure OpenAI Embedding API.

    Expects base_url to be the deployment endpoint, e.g.:
    https://{resource}.openai.azure.com/openai/deployments/{deployment_id}
    """

    MODELS_INFO: dict[str, int | dict[str, int | list[int]]] = {
        "text-embedding-3-large": {
            "default": 3072,
            "dimensions": [256, 512, 1024, 3072],
        },
        "text-embedding-3-small": {"default": 1536, "dimensions": [512, 1536]},
        "text-embedding-ada-002": 1536,
    }

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """
        Generate embeddings using Azure OpenAI.

        Args:
            request: EmbeddingRequest containing texts and parameters

        Returns:
            EmbeddingResponse with embeddings and metadata
        """
        if not self.api_key:
            raise ValueError("API key is required for Azure OpenAI embedding")
        if not self.base_url:
            raise ValueError("Base URL is required for Azure OpenAI embedding")

        headers = {
            "Content-Type": "application/json",
            "api-key": self.api_key,
        }

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

        payload: dict[str, Any] = {
            "input": request.texts,
            "encoding_format": request.encoding_format or "float",
            "model": self.model,
        }

        # Some models support custom dimensions
        if request.dimensions or self.dimensions:
            payload["dimensions"] = request.dimensions or self.dimensions

        logger.debug(
            f"Sending Azure embedding request to {url} with {len(request.texts)} texts"
        )

        async with httpx.AsyncClient(timeout=self.request_timeout) as client:
            try:
                response = await client.post(
                    url, json=payload, headers=headers
                )

                response.raise_for_status()
                data = response.json()
            except httpx.HTTPStatusError as e:
                status_code = (
                    e.response.status_code if e.response else "unknown"
                )
                body = e.response.text if e.response else ""
                logger.error(
                    "Azure API error (HTTP %s): %s", status_code, body
                )
                raise
            except Exception as e:
                logger.error(f"Unexpected error during Azure embedding: {e}")
                raise

        if "data" not in data or not data["data"]:
            raise ValueError(
                f"Invalid API response: missing or empty 'data' field"
            )

        embeddings = [item["embedding"] for item in data["data"]]
        actual_dims = len(embeddings[0]) if embeddings else 0
        # Verify dimensions if requested
        expected_dims = request.dimensions or self.dimensions
        if expected_dims and actual_dims != expected_dims:
            logger.warning(
                f"Dimension mismatch: expected {expected_dims}, got {actual_dims}."
            )

        return EmbeddingResponse(
            embeddings=embeddings,
            model=data.get("model", self.model or "unknown"),
            dimensions=actual_dims,
            usage=data.get("usage", {}),
        )

    def get_model_info(self) -> Dict[str, Any]:
        """
        Return information about the configured model.

        Returns:
            Dictionary with model metadata (name, dimensions, etc.)
        """
        model_name = self.model or ""
        model_info = self.MODELS_INFO.get(model_name, self.dimensions)

        if isinstance(model_info, dict):
            # Help Pylance with types here
            info_dict = cast(dict[str, Any], model_info)
            return {
                "model": model_name,
                "dimensions": info_dict.get("default", self.dimensions),
                "supported_dimensions": info_dict.get("dimensions", []),
                "supports_variable_dimensions": len(
                    cast(list[int], info_dict.get("dimensions", []))
                )
                > 1,
                "provider": "azure",
            }
        else:
            return {
                "model": model_name,
                "dimensions": model_info or self.dimensions,
                "supports_variable_dimensions": False,
                "provider": "azure",
            }
