"""OpenAI-compatible embedding adapter for OpenAI, Azure, HuggingFace, LM Studio, etc."""

import logging
from typing import Any, Dict

import httpx

from .base import BaseEmbeddingAdapter, EmbeddingRequest, EmbeddingResponse
from .error_messages import format_error_message

logger = logging.getLogger(__name__)


class OpenAICompatibleEmbeddingAdapter(BaseEmbeddingAdapter):
    MODELS_INFO = {
        "text-embedding-3-large": {"default": 3072, "dimensions": [256, 512, 1024, 3072]},
        "text-embedding-3-small": {"default": 1536, "dimensions": [512, 1536]},
        "text-embedding-ada-002": 1536,
    }

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        headers = {
            "Content-Type": "application/json",
        }
        if self.api_version:
            headers["api-key"] = self.api_key
        else:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "input": request.texts,
            "model": request.model or self.model,
            "encoding_format": request.encoding_format or "float",
        }

        model_for_request = payload["model"]
        model_info = self.MODELS_INFO.get(model_for_request)
        supports_dimensions = isinstance(model_info, dict)
        if supports_dimensions and (request.dimensions or self.dimensions):
            payload["dimensions"] = request.dimensions or self.dimensions

        base = self.base_url.rstrip("/")
        if self.api_version:
            # Azure OpenAI endpoints are typically already in the correct deployment form
            # and expect /embeddings with api-version query parameter.
            url = f"{base}/embeddings"
        else:
            # OpenAI-compatible servers (OpenAI, LM Studio, vLLM, etc.) typically use /v1/embeddings
            if base.endswith("/v1"):
                url = f"{base}/embeddings"
            else:
                url = f"{base}/v1/embeddings"
        if self.api_version:
            if "?" not in url:
                url += f"?api-version={self.api_version}"
            else:
                url += f"&api-version={self.api_version}"

        logger.debug(f"Sending embedding request to {url} with {len(request.texts)} texts")

        async with httpx.AsyncClient(timeout=self.request_timeout) as client:
            response = await client.post(url, json=payload, headers=headers)

            if response.status_code >= 400:
                error_text = response.text
                logger.error(f"HTTP {response.status_code} response body: {error_text}")

                # Provide helpful error messages for common issues
                if response.status_code == 400:
                    if "invalid" in error_text.lower() or "parameter" in error_text.lower():
                        raise ValueError(format_error_message("invalid_parameters",
                            model=payload['model'], error_text=error_text))
                    else:
                        raise ValueError(format_error_message("invalid_parameters",
                            model=payload.get('model'), error_text=error_text))
                elif response.status_code == 401:
                    raise ValueError(format_error_message("auth_error", base_url=self.base_url))
                elif response.status_code == 404:
                    raise ValueError(format_error_message("model_not_found", model=payload['model']))
                elif response.status_code == 429:
                    raise ValueError(format_error_message("rate_limit"))

            response.raise_for_status()
            data = response.json()

        embeddings = [item.get("embedding", []) for item in (data.get("data") or [])]
        if not embeddings or any(not e for e in embeddings):
            raise ValueError(
                format_error_message(
                    "empty_embedding",
                    model=data.get("model", model_for_request),
                    base_url=self.base_url,
                )
            )

        actual_dims = len(embeddings[0]) if embeddings else 0
        expected_dims = request.dimensions or self.dimensions

        if expected_dims and actual_dims != expected_dims:
            logger.warning(
                f"Dimension mismatch: expected {expected_dims}, got {actual_dims}. "
                f"Model '{data.get('model', model_for_request)}' may not support custom dimensions."
            )

        logger.info(
            f"Successfully generated {len(embeddings)} embeddings "
            f"(model: {data.get('model', model_for_request)}, dimensions: {actual_dims})"
        )

        return EmbeddingResponse(
            embeddings=embeddings,
            model=data.get("model", model_for_request),
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
