# -*- coding: utf-8 -*-
"""
OpenAI-compatible embedding adapter.

Supports OpenAI, Azure OpenAI, HuggingFace, and LM Studio-style endpoints.
"""

import logging
import os
from typing import Any, cast
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

import httpx

from .base import BaseEmbeddingAdapter, EmbeddingRequest, EmbeddingResponse

logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 100


def _chunk_texts(texts: list[str], batch_size: int) -> list[list[str]]:
    """
    Split a list of texts into consecutive batches of up to `batch_size` items.
    
    Parameters:
    	texts (list[str]): The input texts to partition.
    	batch_size (int): Maximum number of texts per batch; if <= 0, returns a single batch containing all texts.
    
    Returns:
    	list[list[str]]: A list of text batches, each batch being a list of strings.
    """
    if batch_size <= 0:
        return [texts]

    return [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]


def _merge_usage(total: dict[str, int], usage: dict[str, Any]) -> None:
    """
    Accumulate integer usage counters from a single response into a running totals dictionary.
    
    Parameters:
        total (dict[str, int]): Mutable dictionary of aggregated usage counters; keys map to integer totals.
        usage (dict[str, Any]): Usage payload from a single response; only entries with integer values are added to `total`.
    """
    for key, value in usage.items():
        if isinstance(value, int):
            total[key] = total.get(key, 0) + value


class OpenAICompatibleEmbeddingAdapter(BaseEmbeddingAdapter):
    """
    Adapter for OpenAI-compatible embedding endpoints.

    Args:
        None.

    Returns:
        None.
    """

    MODELS_INFO: dict[str, int | dict[str, int | list[int]]] = {
        "text-embedding-3-large": {
            "default": 3072,
            "dimensions": [256, 512, 1024, 3072],
        },
        "text-embedding-3-small": {"default": 1536, "dimensions": [512, 1536]},
        "text-embedding-ada-002": 1536,
    }

    def __init__(self, config: dict[str, Any]):
        """
        Create an OpenAICompatibleEmbeddingAdapter configured from the provided settings.
        
        Parameters:
            config (dict[str, Any]): Adapter configuration. Recognized keys:
                - "batch_size" (int, optional): Number of texts per request. Defaults to DEFAULT_BATCH_SIZE.
        """
        super().__init__(config)
        self.batch_size = config.get("batch_size", DEFAULT_BATCH_SIZE)

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """
        Generate embeddings from an OpenAI-compatible embedding endpoint.
        
        Parameters:
            request (EmbeddingRequest): Contains input texts and optional parameters such as
                model, dimensions, and encoding_format.
        
        Returns:
            EmbeddingResponse: Contains the list of embeddings, the model used (may be updated
            from the API response), the actual embedding dimensionality, and aggregated usage
            counters across batched requests.
        
        Raises:
            ValueError: If required configuration (base URL or model) is missing or if the API
                response is missing expected fields (`data` or per-item `embedding`).
            httpx.HTTPError: If an HTTP request fails.
        """
        if not self.base_url:
            raise ValueError("Base URL is required for OpenAI-compatible embedding")

        model = request.model or self.model
        if not model:
            raise ValueError("Model is required for OpenAI-compatible embedding")

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
            "model": model,
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

        embeddings: list[list[float]] = []
        usage_totals: dict[str, int] = {}
        response_model = model

        # Check if SSL verification should be disabled for testing
        if not hasattr(self, "_ssl_warning_logged"):
            disable_ssl = os.getenv("DISABLE_SSL_VERIFY", "").lower()
            self._disable_ssl_verify = disable_ssl in ("true", "1", "yes")
            if self._disable_ssl_verify:
                logger.warning(
                    "SSL verification has been disabled via the DISABLE_SSL_VERIFY "
                    "environment variable. This should only be used in testing and "
                    "never in production."
                )
                self._ssl_warning_logged = True  # Ensure warning is logged only once

        async with httpx.AsyncClient(
            timeout=self.request_timeout, verify=not self._disable_ssl_verify
        ) as client:
            for batch in _chunk_texts(request.texts, self.batch_size):
                payload["input"] = batch
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

                for item in data["data"]:
                    if "embedding" not in item:
                        raise ValueError("Invalid API response: item missing 'embedding' field")
                    embeddings.append(item["embedding"])
                response_model = data.get("model", response_model)
                _merge_usage(usage_totals, data.get("usage", {}))

        actual_dims = len(embeddings[0]) if embeddings else 0
        expected_dims = request.dimensions or self.dimensions

        if expected_dims and actual_dims != expected_dims:
            logger.warning(
                "Dimension mismatch: expected %d, got %d. Model '%s' may not "
                "support custom dimensions.",
                expected_dims,
                actual_dims,
                response_model,
            )

        logger.info(
            "Successfully generated %d embeddings (model: %s, dimensions: %d)",
            len(embeddings),
            response_model,
            actual_dims,
        )

        return EmbeddingResponse(
            embeddings=embeddings,
            model=response_model,
            dimensions=actual_dims,
            usage=usage_totals,
        )

    def get_model_info(self) -> dict[str, Any]:
        """
        Provide metadata about the adapter's configured model.
        
        Returns:
            dict: Metadata with the following keys:
                - model (str): Configured model name (empty string if unset).
                - dimensions (int | None): Effective embedding dimensionality for the model or None if unknown.
                - supported_dimensions (list[int]): List of supported dimensionalities reported for the model.
                - supports_variable_dimensions (bool): `true` when more than one supported dimension is available.
                - provider (str): Provider identifier ("openai_compatible").
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