# -*- coding: utf-8 -*-
"""
Base Embedding Adapter
=======================

Abstract base class for all embedding adapters.
Defines the strict contract for system hardening.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Sequence, Dict, Optional


@dataclass(frozen=True, slots=True)
class EmbeddingRequest:
    """
    Standard immutable embedding request structure.

    Args:
        texts: Sequence of strings (Immutable to prevent side-effects).
        model: Provider-specific model identifier.
        dimensions: Optional output vector size.
        input_type: Task hint (search_query, document, etc).
    """

    texts: Sequence[str]
    model: str
    dimensions: Optional[int] = None
    input_type: Optional[str] = None
    encoding_format: str = "float"
    truncate: bool = True
    normalized: bool = True
    late_chunking: bool = False


@dataclass(frozen=True, slots=True)
class EmbeddingResponse:
    """Standard standardized embedding response structure."""

    embeddings: list[list[float]]
    model: str
    dimensions: int
    usage: Dict[str, Any] = field(default_factory=dict)


class BaseEmbeddingAdapter(ABC):
    """
    Abstract base class for embedding adapters.
    Enforces defensive checks and lifecycle management.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.api_key: Optional[str] = config.get("api_key")
        self.base_url: Optional[str] = config.get("base_url")
        self.api_version: Optional[str] = config.get("api_version")
        self.model: Optional[str] = config.get("model")
        self.dimensions: Optional[int] = config.get("dimensions")
        self.request_timeout: int = int(config.get("request_timeout", 30))

        # Internal batch limit to prevent payload overflows.
        self.max_batch_size: int = int(config.get("max_batch_size", 100))

    def _validate_texts(self, texts: Sequence[str]) -> None:
        """Perform pre-flight validation on input sequences."""
        if not texts:
            raise ValueError("Embedding request contains no texts.")

        if len(texts) > self.max_batch_size:
            raise ValueError(f"Batch size {len(texts)} exceeds limit of {self.max_batch_size}")

        if any(not isinstance(t, str) for t in texts):
            raise ValueError("Embedding input sequence contains non-string elements.")

    @abstractmethod
    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Execute embedding generation for the provided request."""
        ...

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Expose model metadata for telemetry and introspection."""
        ...

    async def close(self) -> None:
        """Optional hook for cleaning up network resources."""
        ...
