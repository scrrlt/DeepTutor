# -*- coding: utf-8 -*-
"""
Azure OpenAI Embedding Adapter
==============================

Hardened adapter for Azure OpenAI Embedding API using the official SDK.
Enforces strict dimension contracts, numeric integrity sampling,
and audit-grade telemetry.
"""

import logging
import math
from typing import Any, Dict, Optional, Tuple

import openai

from .base import BaseEmbeddingAdapter, EmbeddingRequest, EmbeddingResponse

logger = logging.getLogger(__name__)


class AzureEmbeddingAdapter(BaseEmbeddingAdapter):
    """
    Adapter for Azure OpenAI Service.

    Implements defense-in-depth for vector generation to prevent silent
    index corruption in high-volume retrieval pipelines.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)

        if not self.api_key:
            raise ValueError("Azure API key missing")
        if not self.base_url:
            raise ValueError("Azure endpoint (base_url) missing")
        if not self.model:
            raise ValueError("Azure deployment name (model) missing")

        # Configuration Policy: Strict mode treats dimension drift as fatal.
        self.strict_dimensions: bool = config.get("strict_dimensions", True)
        self._api_version: str = self.api_version or "2023-05-15"

        self._client = openai.AsyncAzureOpenAI(
            api_key=self.api_key,
            azure_endpoint=self.base_url,
            api_version=self._api_version,
            timeout=self.request_timeout,
        )

    async def __aenter__(self) -> "AzureEmbeddingAdapter":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()

    async def close(self) -> None:
        """Release the underlying SDK transport to prevent connection leaks."""
        await self._client.close()

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Generate embeddings with multi-stage integrity validation."""
        self._validate_texts(request.texts)

        # 1. Pre-flight: Prevent requests that conflict with adapter contract.
        if (
            self.strict_dimensions
            and request.dimensions
            and self.dimensions
            and request.dimensions != self.dimensions
        ):
            logger.error(
                "Pre-flight rejection: Dimension mismatch. Config: %d, Req: %d",
                self.dimensions,
                request.dimensions,
            )
            raise ValueError(
                f"Requested dimensions ({request.dimensions}) conflict with "
                f"adapter contract ({self.dimensions})"
            )

        target_dims = request.dimensions or self.dimensions

        try:
            response = await self._client.embeddings.create(
                input=request.texts,
                model=self.model,
                dimensions=target_dims,
                encoding_format=request.encoding_format,
            )

            if not response.data:
                raise ValueError("Azure API returned empty response data.")

            embeddings = [item.embedding for item in response.data]

            # 2. Integrity: Sampling numeric health.
            if not embeddings or not isinstance(embeddings[0], list):
                raise ValueError("Malformed embedding structure returned from SDK.")

            actual_dims = len(embeddings[0])
            if not all(
                isinstance(v, (float, int)) and math.isfinite(v) for v in embeddings[0][:10]
            ):
                raise ValueError("Non-numeric or non-finite values detected in vector.")

            # 3. Post-flight: Guard against silent server-side drift.
            if self.strict_dimensions and target_dims and actual_dims != target_dims:
                logger.critical(
                    "DIMENSION DRIFT: Expected %d, got %d. Deployment: %s",
                    target_dims,
                    actual_dims,
                    self.model,
                )
                raise ValueError(f"Dimension mismatch: expected {target_dims}, got {actual_dims}")

            usage = response.usage.model_dump() if response.usage else {}
            usage.update(
                {
                    "api_version": self._api_version,
                    "azure_endpoint": self.base_url,
                    "provider": "azure",
                    "strict_mode": self.strict_dimensions,
                }
            )

            return EmbeddingResponse(
                embeddings=embeddings,
                model=response.model,
                dimensions=actual_dims,
                usage=usage,
            )

        except (openai.APIConnectionError, openai.APITimeoutError) as e:
            logger.error("Azure Network/Timeout Failure: %s", e)
            raise
        except openai.APIStatusError as e:
            logger.error("Azure API Error [%d]: %s", e.status_code, e.message)
            raise
        except Exception as e:
            logger.exception("Uncaught exception in Azure embedding pipeline: %s", e)
            raise

    def get_model_info(self) -> Dict[str, Any]:
        """Expose model metadata and hardened capabilities."""
        return {
            "model": self.model,
            "dimensions": self.dimensions,
            "provider": "azure",
            "strict_mode": self.strict_dimensions,
            "dimension_contract": self.dimensions,
            "capabilities": {
                "variable_dimensions": True,
                "batch_support": True,
                "context_manager": True,
            },
        }
