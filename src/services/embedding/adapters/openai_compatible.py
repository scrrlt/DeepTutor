"""OpenAI-compatible embedding adapter for OpenAI, Azure, HuggingFace, LM Studio, etc."""

import logging
from typing import Any, Dict, Union

import httpx

from .base import BaseEmbeddingAdapter, EmbeddingRequest, EmbeddingResponse

logger = logging.getLogger(__name__)


class OpenAICompatibleEmbeddingAdapter(BaseEmbeddingAdapter):
    MODELS_INFO: Dict[str, Union[int, Dict[str, Any]]] = {
        "text-embedding-3-large": {"default": 3072, "dimensions": [256, 512, 1024, 3072]},
        "text-embedding-3-small": {"default": 1536, "dimensions": [512, 1536]},
        "text-embedding-ada-002": 1536,
    }

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        headers = {
            "Content-Type": "application/json",
        }
        if self.api_version:
            headers["api-key"] = self.api_key  # type: ignore
        else:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload: Dict[str, Any] = {
            "input": request.texts,
            "model": request.model or self.model,
            "encoding_format": request.encoding_format or "float",
        }

        req_model = request.model or self.model
        if req_model is None:
            raise ValueError("Model must be specified in request or configuration")

        model_info = self.get_model_info()
        if (request.dimensions or self.dimensions) and model_info.get(  # type: ignore
            "supports_variable_dimensions", False
        ):
            payload["dimensions"] = request.dimensions or self.dimensions

        url = f"{(self.base_url or '').rstrip('/')}/embeddings"
        if self.api_version:
            if "?" not in url:
                url += f"?api-version={self.api_version}"
            else:
                url += f"&api-version={self.api_version}"

        logger.debug(f"Sending embedding request to {url} with {len(request.texts)} texts")

        async with httpx.AsyncClient(timeout=self.request_timeout) as client:
            response = await client.post(url, json=payload, headers=headers)  # type: ignore

            if response.status_code >= 400:
                logger.error(f"HTTP {response.status_code} response body: {response.text}")

            response.raise_for_status()
            data = response.json()

        resp_model = str(data.get("model", req_model))

        embeddings = [item["embedding"] for item in data["data"]]

        actual_dims = len(embeddings[0]) if embeddings else 0
        expected_dims = request.dimensions or self.dimensions

        if expected_dims and actual_dims != expected_dims:
            logger.warning(
                f"Dimension mismatch: expected {expected_dims}, got {actual_dims}. "
                f"Model '{resp_model}' may not support custom dimensions."
            )

        logger.info(
            f"Successfully generated {len(embeddings)} embeddings "
            f"(model: {resp_model}, dimensions: {actual_dims})"
        )

        return EmbeddingResponse(
            embeddings=embeddings,
            model=resp_model,
            dimensions=actual_dims,
            usage=data.get("usage", {}),
        )

    def get_model_info(self) -> Dict[str, Any]:
        model_info = self.MODELS_INFO.get(self.model, self.dimensions)

        if isinstance(model_info, dict):
            return {
                "model": self.model,
                "dimensions": model_info.get("default", self.dimensions),
                "supported_dimensions": model_info.get("dimensions", []),
                "supports_variable_dimensions": len(model_info.get("dimensions", [])) > 1,
                "provider": "openai_compatible",
            }
        else:
            return {
                "model": self.model,
                "dimensions": model_info or self.dimensions,
                "supports_variable_dimensions": False,
                "provider": "openai_compatible",
            }
