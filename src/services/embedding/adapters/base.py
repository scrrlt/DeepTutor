"""
Base Embedding Adapter
=======================

Abstract base class for all embedding adapters.
Defines the contract that all embedding providers must implement.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class EmbeddingRequest:
    """
    Standard embedding request structure.

    Provider-agnostic request format. Different providers interpret fields differently:

    Args:
        texts: List of texts to embed
        model: Model name to use
        dimensions: Embedding vector dimensions (optional)
        input_type: Input type hint for task-aware embeddings (optional)
            - Cohere: Maps to 'input_type' ("search_document", "search_query", "classification", "clustering")
            - Jina: Maps to 'task' ("retrieval.passage", "retrieval.query", etc.)
            - OpenAI/Ollama: Ignored
        encoding_format: Output format ("float" or "base64", default: "float")
        truncate: Whether to truncate texts that exceed max tokens (default: True)
        normalized: Whether to return L2-normalized embeddings (Jina/Ollama only)
        late_chunking: Enable late chunking for long context (Jina v3 only)
    """

    texts: list[str]
    model: str
    dimensions: int | None = None
    input_type: str | None = None
    encoding_format: str | None = "float"
    truncate: bool | None = True
    normalized: bool | None = True
    late_chunking: bool | None = False


@dataclass
class EmbeddingResponse:
    """Standard embedding response structure."""

    embeddings: list[list[float]]
    model: str
    dimensions: int
    usage: dict[str, Any]


class BaseEmbeddingAdapter(ABC):
    """
    Base class for all embedding adapters.

    Each adapter implements the specific API interface for a provider
    (OpenAI, Cohere, Ollama, etc.) while exposing a unified interface.
    """

    def _validate_texts(self, texts: list[str]) -> None:
        """
        Validate that texts list is not empty.

        Args:
            texts: List of texts to validate

        Raises:
            ValueError: If texts list is empty
        """
        if not texts:
            raise ValueError("Embedding request requires at least one text")

    def __init__(self, config: dict[str, Any]):
        """
        Initialize the adapter with configuration.

        Args:
            config: Dictionary containing:
                - api_key: API authentication key (optional for local)
                - base_url: API endpoint URL
                - api_version: API version (optional, provider-specific)
                - model: Model name to use
                - dimensions: Embedding vector dimensions
                - request_timeout: Request timeout in seconds
        """
        self.api_key: str | None = config.get("api_key")
        self.base_url: str | None = config.get("base_url")
        self.api_version: str | None = config.get("api_version")
        self.model: str | None = config.get("model")
        self.dimensions: int | None = config.get("dimensions")
        self.request_timeout: int = config.get("request_timeout", 30)

    @abstractmethod
    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """
        Generate embeddings for a list of texts.

        Args:
            request: EmbeddingRequest with texts and parameters

        Returns:
            EmbeddingResponse with embeddings and metadata

        Raises:
            httpx.HTTPError: If the API request fails
        """
        pass

    @abstractmethod
    def get_model_info(self) -> dict[str, Any]:
        """
        Return information about the configured model.

        Returns:
            Dictionary with model metadata (name, dimensions, etc.)
        """
        pass
