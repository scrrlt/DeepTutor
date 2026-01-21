# -*- coding: utf-8 -*-
"""
Embedding Configuration
=======================

Configuration management for embedding services.
Simplified version - loads from unified config service or falls back to .env.
"""

from dataclasses import dataclass
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load environment variables
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env.local", override=False)
load_dotenv(PROJECT_ROOT / "DeepTutor.env", override=False)
load_dotenv(PROJECT_ROOT / ".env", override=False)


@dataclass
class EmbeddingConfig:
    """Embedding configuration dataclass."""

    model: str
    api_key: str
    base_url: str | None = None
    binding: str = "openai"
    api_version: str | None = None
    dim: int | None = None
    max_tokens: int = 8192
    request_timeout: int = 30
    input_type: str | None = None  # For task-aware embeddings

    # Optional provider-specific settings
    encoding_format: str = "float"
    normalized: bool = True
    truncate: bool = True
    late_chunking: bool = False


def _strip_value(value: str | None) -> str | None:
    """Remove leading/trailing whitespace and quotes from string."""
    if value is None:
        return None
    return value.strip().strip("\"'")


def _to_int(value: str | None, default: int | None) -> int | None:
    """Convert environment variable to int, fallback to default value on failure."""
    try:
        return int(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def _to_bool(value: str | None, default: bool) -> bool:
    """Convert environment variable to bool."""
    if value is None:
        return default
    return value.lower() in ("true", "1", "yes", "on")


def _get_default_dimensions(model: str) -> int:
    """
    Get default embedding dimensions for known models.

    Args:
        model: Model name (case-insensitive)

    Returns:
        Default dimensions for the model
    """
    model_lower = model.lower()

    # OpenAI models
    if "text-embedding-ada-002" in model_lower:
        return 1536
    elif "text-embedding-3-small" in model_lower:
        return 1536
    elif "text-embedding-3-large" in model_lower:
        return 3072

    # Anthropic models (if they add embeddings)
    # Add more as needed

    # Default fallback
    return 3072


def get_embedding_config() -> EmbeddingConfig:
    """
    Load embedding configuration.

    Priority:
    1. Active configuration from unified config service
    2. Environment variables (.env)

    Returns:
        EmbeddingConfig: Configuration dataclass

    Raises:
        ValueError: If required configuration is missing
    """
    # 1. Try to get active config from unified config service
    try:
        from src.services.config import get_active_embedding_config

        config = get_active_embedding_config()
        if config and config.get("model"):
            dim = config.get("dimensions")
            if dim is None:
                dim = _get_default_dimensions(config["model"])
            return EmbeddingConfig(
                binding=config.get("provider", "openai"),
                model=config["model"],
                api_key=config.get("api_key", ""),
                base_url=config.get("base_url"),
                api_version=config.get("api_version"),
                dim=dim,
            )
    except ImportError:
        # Unified config service not yet available, fall back to env
        pass
    except Exception as e:
        logger.warning("Failed to load from unified config: %s", e)

    # 2. Fallback to environment variables
    binding = _strip_value(os.getenv("EMBEDDING_BINDING", "openai"))
    model = _strip_value(os.getenv("EMBEDDING_MODEL"))
    api_key = _strip_value(os.getenv("EMBEDDING_API_KEY"))
    base_url = _strip_value(os.getenv("EMBEDDING_HOST"))
    api_version = _strip_value(os.getenv("EMBEDDING_API_VERSION"))
    dim_str = _strip_value(os.getenv("EMBEDDING_DIMENSION"))

    # Strict mode: Model is required
    if not model:
        model = "text-embedding-3-large"  # Default fallback model
        dim = 3072  # Use default for this model

    # Check if API key is required
    # Local providers (Ollama, LM Studio) don't need API keys
    providers_without_key = ["ollama", "lm_studio"]
    requires_key = binding not in providers_without_key

    if requires_key and not api_key:
        raise ValueError(
            "EMBEDDING_API_KEY not set. Please configure it in .env file or add a configuration in Settings"
        )
    if not base_url:
        raise ValueError(
            "EMBEDDING_HOST not set. Please configure it in .env file or add a configuration in Settings"
        )

    # Get optional configuration

    # If dim not specified, use model-appropriate default
    if dim is None:
        dim = _get_default_dimensions(model)
    # Override for known models to ensure correctness
    if model == "text-embedding-3-small":
        dim = 1536
    elif model == "text-embedding-3-large":
        dim = 3072
    max_tokens = _to_int(_strip_value(os.getenv("EMBEDDING_MAX_TOKENS")), 8192)
    request_timeout = _to_int(
        _strip_value(os.getenv("EMBEDDING_REQUEST_TIMEOUT")), 30
    )
    input_type = _strip_value(os.getenv("EMBEDDING_INPUT_TYPE"))  # Optional

    # Provider-specific optional settings
    encoding_format = (
        _strip_value(os.getenv("EMBEDDING_ENCODING_FORMAT")) or "float"
    )
    normalized = _to_bool(
        _strip_value(os.getenv("EMBEDDING_NORMALIZED")), True
    )
    truncate = _to_bool(_strip_value(os.getenv("EMBEDDING_TRUNCATE")), True)
    late_chunking = _to_bool(
        _strip_value(os.getenv("EMBEDDING_LATE_CHUNKING")), False
    )

    return EmbeddingConfig(
        binding=binding or "openai",
        model=model,
        api_key=api_key or "",
        base_url=base_url,
        api_version=api_version,
        dim=dim,
        max_tokens=max_tokens,
        request_timeout=request_timeout,
        input_type=input_type,
        encoding_format=encoding_format,
        normalized=normalized,
        truncate=truncate,
        late_chunking=late_chunking,
    )
