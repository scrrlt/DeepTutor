# -*- coding: utf-8 -*-
"""
LLM Configuration
=================

Configuration management for LLM services.
Simplified version - loads from unified config service or falls back to .env.
"""

from dataclasses import dataclass
import logging
import os
from pathlib import Path
import re
from typing import Optional

from dotenv import load_dotenv

from .exceptions import LLMConfigError

logger = logging.getLogger(__name__)

# Load environment variables
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
load_dotenv(PROJECT_ROOT / "DeepTutor.env", override=False)
load_dotenv(PROJECT_ROOT / ".env", override=False)
load_dotenv(PROJECT_ROOT / ".env.local", override=False)


@dataclass
class LLMConfig:
    """LLM configuration dataclass."""

    model: str
    api_key: str
    base_url: Optional[str] = None
    binding: str = "openai"
    api_version: Optional[str] = None
    max_tokens: int = 4096
    temperature: float = 0.7


def initialize_environment():
    """
    Ensure OpenAI-compatible environment variables are populated from LLM_* variables for compatibility.
    
    If the configured binding is "openai", "azure_openai", or "gemini", sets OPENAI_API_KEY from LLM_API_KEY and OPENAI_BASE_URL from LLM_HOST when those OPENAI_* variables are not already present. Intended to be called during application startup to provide compatibility with code paths that read OPENAI_API_KEY/OPENAI_BASE_URL directly.
    """
    binding = _strip_value(os.getenv("LLM_BINDING")) or "openai"
    api_key = _strip_value(os.getenv("LLM_API_KEY"))
    base_url = _strip_value(os.getenv("LLM_HOST"))

    # Only set env vars for OpenAI-compatible bindings
    if binding in ("openai", "azure_openai", "gemini"):
        if api_key and not os.getenv("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = api_key
            logger.debug("Set OPENAI_API_KEY env var (LightRAG compatibility)")

        if base_url and not os.getenv("OPENAI_BASE_URL"):
            os.environ["OPENAI_BASE_URL"] = base_url
            logger.debug(f"Set OPENAI_BASE_URL env var to {base_url}")


def _strip_value(value: Optional[str]) -> Optional[str]:
    """
    Strip surrounding whitespace and surrounding single or double quotes from a string.
    
    Returns:
        `None` if `value` is `None`; otherwise the input string with leading and trailing whitespace and any surrounding single or double quotes removed.
    """
    if value is None:
        return None
    return value.strip().strip("\"'")


def _get_llm_config_from_env() -> LLMConfig:
    """
    Load LLM configuration from environment variables.
    
    Reads LLM_BINDING (defaults to "openai"), LLM_MODEL, LLM_API_KEY, LLM_HOST, and LLM_API_VERSION and returns an LLMConfig populated from those values. If LLM_API_KEY is missing the returned config will contain an empty string for api_key.
    
    Returns:
        LLMConfig: Configuration constructed from environment variables.
    
    Raises:
        LLMConfigError: If LLM_MODEL or LLM_HOST is not set.
    """
    binding = _strip_value(os.getenv("LLM_BINDING")) or "openai"
    model = _strip_value(os.getenv("LLM_MODEL"))
    api_key = _strip_value(os.getenv("LLM_API_KEY"))
    base_url = _strip_value(os.getenv("LLM_HOST"))
    api_version = _strip_value(os.getenv("LLM_API_VERSION"))

    # Validate required configuration
    if not model:
        raise LLMConfigError(
            "LLM_MODEL not set, please configure it in .env file or add a configuration in Settings"
        )
    if not base_url:
        raise LLMConfigError(
            "LLM_HOST not set, please configure it in .env file or add a configuration in Settings"
        )

    return LLMConfig(
        binding=binding,
        model=model,
        api_key=api_key or "",
        base_url=base_url,
        api_version=api_version,
    )


def get_llm_config() -> LLMConfig:
    """
    Load LLM configuration.

    Priority:
    1. Active configuration from unified config service
    2. Environment variables (.env)

    Returns:
        LLMConfig: Configuration dataclass

    Raises:
        LLMConfigError: If required configuration is missing
    """
    # 1. Try to get active config from unified config service
    try:
        from src.services.config import get_active_llm_config

        config = get_active_llm_config()
        if config:
            return LLMConfig(
                binding=config.get("provider") or "openai",
                model=config["model"],
                api_key=config.get("api_key", ""),
                base_url=config.get("base_url"),
                api_version=config.get("api_version"),
            )
    except ImportError:
        # Unified config service not yet available, fall back to env
        pass
    except Exception as e:
        logger.warning(f"Failed to load from unified config: {e}")

    # 2. Fallback to environment variables
    return _get_llm_config_from_env()


async def get_llm_config_async() -> LLMConfig:
    """
    Provide an async-compatible entry point that returns the active LLM configuration.
    
    This exists for API consistency in asynchronous code paths.
    
    Returns:
        LLMConfig: The active LLM configuration.
    """
    return get_llm_config()


def uses_max_completion_tokens(model: str) -> bool:
    """
    Determine whether a model uses the `max_completion_tokens` parameter instead of `max_tokens`.
    
    Parameters:
        model (str): Model name to evaluate.
    
    Returns:
        `true` if the model uses `max_completion_tokens`, `false` otherwise.
    """
    model_lower = model.lower()

    # Models that require max_completion_tokens:
    # - o1, o3 series (reasoning models)
    # - gpt-4o series
    # - gpt-5.x and later
    patterns = [
        r"^o[13]",  # o1, o3 models
        r"^gpt-4o",  # gpt-4o models
        r"^gpt-[5-9]",  # gpt-5.x and later
        r"^gpt-\d{2,}",  # gpt-10+ (future proofing)
    ]

    for pattern in patterns:
        if re.match(pattern, model_lower):
            return True

    return False


def get_token_limit_kwargs(model: str, max_tokens: int) -> dict[str, int]:
    """
    Choose the correct token-limit parameter name for the given model.
    
    Parameters:
        model (str): Model identifier used to determine which token limit key to use.
        max_tokens (int): Token limit value to assign.
    
    Returns:
        dict[str, int]: A mapping containing either `{"max_completion_tokens": <value>}` for models that use completion-specific limits or `{"max_tokens": <value>}` for models that use the general token limit.
    """
    if uses_max_completion_tokens(model):
        return {"max_completion_tokens": max_tokens}
    return {"max_tokens": max_tokens}


__all__ = [
    "LLMConfig",
    "get_llm_config",
    "get_llm_config_async",
    "uses_max_completion_tokens",
    "get_token_limit_kwargs",
]