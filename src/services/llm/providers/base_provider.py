"""
Base LLM Provider - Unified interface and configuration.
"""

from abc import ABC, abstractmethod
import os
import asyncio
import random
import logging
from typing import Any, Dict, Callable

logger = logging.getLogger(__name__)


class BaseLLMProvider(ABC):
    """Base class for all LLM providers with unified config."""

    def __init__(self, config):
        """Initialize with config object."""
        self.config = config
        self.provider_name = config.provider_name
        self.api_key = config.api_key
        self.base_url = config.base_url
        self.deployment = getattr(config, 'deployment', None) or os.getenv(f"{config.provider_name.upper()}_DEPLOYMENT_NAME")
        self.api_version = config.api_version

        if not self.api_key:
            raise ValueError(f"Missing API key for {config.provider_name}")

    def _default_base_url(self) -> str:
        """Default base URL for the provider."""
        return ""

    @property
    def extra_headers(self) -> Dict[str, str]:
        """Provider-specific headers."""
        # Default to OpenAI style; override in subclasses for Azure, etc.
        return {"Authorization": f"Bearer {self.api_key}"}

    def resolve_model(self, requested_model: str) -> str:
        """Resolve model name/deployment ID."""
        # Default: return as-is
        return requested_model

    @abstractmethod
    async def complete(self, prompt: str, **kwargs) -> str:
        """Complete a prompt."""
        pass

    def calculate_cost(self, usage: Dict[str, Any]) -> float:
        """Calculate cost (can be overridden)."""
        # Default implementation
        return 0.0

    def validate_config(self):
        """Validate provider-specific configuration."""
        pass  # Override in subclasses

    async def execute_with_retry(self, func: Callable, *args, max_retries=3, **kwargs):
        """Executes a function with exponential backoff retry logic."""
        for attempt in range(max_retries + 1):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                error_str = str(e).lower()
                is_retriable = (
                    "429" in error_str or "rate limit" in error_str or
                    "quota" in error_str or "500" in error_str or
                    "503" in error_str or "timeout" in error_str
                )

                if attempt >= max_retries or not is_retriable:
                    raise e

                delay = (1.5 ** attempt) + (random.random() * 0.5)
                logger.warning(f"LLM call failed ({e}). Retrying in {delay:.2f}s...")
                await asyncio.sleep(delay)
