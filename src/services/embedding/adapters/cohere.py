"""Cohere Embedding Adapter for v1 and v2 API."""

import logging
from typing import Any, Dict

import httpx

from .base import BaseEmbeddingAdapter, EmbeddingRequest, EmbeddingResponse

logger = logging.getLogger(__name__)


class CohereEmbeddingAdapter(BaseEmbeddingAdapter):
    """Adapter for Cohere Embed API (v1 and v2)."""

    MODELS_INFO: Dict[str, Dict[str, Any]] = {
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
        if not self.api_key:
            raise ValueError("API key is required for Cohere embeddings")
        if not self.base_url:
            raise ValueError("Base URL is required for Cohere embeddings")

        base_url = self.base_url.rstrip("/")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        model_name = request.model or self.model
        if model_name is None:
            raise ValueError("Model must be specified for Cohere embeddings")
        model_info = self.MODELS_INFO.get(model_name, {})
        api_version = model_info.get("api_version", "v2")
        dimension = request.dimensions or self.dimensions

        input_type = request.input_type or "search_document"

        payload: Dict[str, Any]
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

        url = f"{base_url}/{api_version}/embed"

        logger.debug(f"Sending embedding request to {url} with {len(request.texts)} texts")

        async with httpx.AsyncClient(timeout=self.request_timeout) as client:
            response = await client.post(url, json=payload, headers=headers)

            if response.status_code >= 400:
                logger.error(f"HTTP {response.status_code} response body: {response.text}")

            response.raise_for_status()
            data = response.json()

        if api_version == "v1":
            embeddings = data["embeddings"]
        else:
            embeddings = data["embeddings"]["float"]

        actual_dims = len(embeddings[0]) if embeddings else 0
        expected_dims = request.dimensions or self.dimensions

        if expected_dims and actual_dims != expected_dims:
            logger.warning(f"Dimension mismatch: expected {expected_dims}, got {actual_dims}")

        logger.info(
            f"Successfully generated {len(embeddings)} embeddings "
            f"(model: {data.get('model', self.model)}, dimensions: {actual_dims})"
        )

        return EmbeddingResponse(
            embeddings=embeddings,
            model=data.get("model", self.model),
            dimensions=actual_dims,
            usage=data.get("meta", {}).get("billed_units", {}),
        )

    def get_model_info(self) -> Dict[str, Any]:
        model_name = self.model or ""
        model_info = self.MODELS_INFO.get(model_name, {})
        dimensions_list = model_info.get("dimensions", [])
        return {
            "model": model_name,
            "dimensions": model_info.get("default", self.dimensions) or 0,
            "supports_variable_dimensions": len(dimensions_list) > 1
            if isinstance(dimensions_list, list)
            else False,
            "provider": "cohere",
        }
