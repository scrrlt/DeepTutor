"""
Embedding Configuration
=======================

Configuration management for embedding services.
"""

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load environment variables
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
load_dotenv(PROJECT_ROOT / "DeepTutor.env", override=False)
load_dotenv(PROJECT_ROOT / ".env", override=False)


@dataclass
class EmbeddingConfig:
    """Embedding configuration dataclass."""
    
    model: str
    api_key: str
    base_url: Optional[str] = None
    binding: str = "openai"
    dim: int = 3072
    max_tokens: int = 8192


def _strip_value(value: Optional[str]) -> Optional[str]:
    """Remove leading/trailing whitespace and quotes from string."""
    if value is None:
        return None
    return value.strip().strip("\"'")


def _to_int(value: Optional[str], default: int) -> int:
    """Convert environment variable to int, fallback to default value on failure."""
    try:
        return int(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def get_embedding_config() -> EmbeddingConfig:
    """
    Load embedding configuration from environment variables.

    Returns:
        EmbeddingConfig: Configuration dataclass

    Raises:
        ValueError: If required configuration is missing
    """
    binding = _strip_value(os.getenv("EMBEDDING_BINDING", "openai"))
    model = _strip_value(os.getenv("EMBEDDING_MODEL"))
    api_key = _strip_value(os.getenv("EMBEDDING_BINDING_API_KEY"))
    base_url = _strip_value(os.getenv("EMBEDDING_BINDING_HOST"))

    # Strict mode: All model configuration must come from .env
    if not model:
        raise ValueError("Error: EMBEDDING_MODEL not set, please configure it in .env file")

    # Check if API key is required
    requires_key = os.getenv("EMBEDDING_API_KEY_REQUIRED", "true").lower() == "true"

    if requires_key and not api_key:
        raise ValueError(
            "Error: EMBEDDING_BINDING_API_KEY not set, please configure it in .env file"
        )
    if not base_url:
        raise ValueError("Error: EMBEDDING_BINDING_HOST not set, please configure it in .env file")

    # Get optional configuration
    dim = _to_int(_strip_value(os.getenv("EMBEDDING_DIM")), 3072)
    max_tokens = _to_int(_strip_value(os.getenv("EMBEDDING_MAX_TOKENS")), 8192)

    return EmbeddingConfig(
        binding=binding,
        model=model,
        api_key=api_key,
        base_url=base_url,
        dim=dim,
        max_tokens=max_tokens,
    )

