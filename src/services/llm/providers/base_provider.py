"""
Base LLM Provider - Unified interface and configuration.
"""

from abc import ABC, abstractmethod
import asyncio
import logging
import os
import random
from typing import Any, Dict, Callable, AsyncGenerator

from ..types import TutorResponse, TutorStreamChunk, AsyncStreamGenerator
from ..exceptions import (
    LLMBaseError,
    ProviderQuotaExceededError,
    ProviderContextWindowError,
    LLMConfigurationError,
    LLMAuthenticationError,
    LLMRateLimitError,
    LLMTimeoutError,
    LLMAPIError,
)

logger = logging.getLogger(__name__)


class BaseLLMProvider(ABC):
    """Base class for all LLM providers with unified config."""

    def __init__(self, config):
        """Initialize with config object."""
        self.config = config
        self.provider_name = config.provider_name
        self.api_key = config.api_key
        self.base_url = config.base_url
        self.deployment = getattr(config, "deployment", None) or os.getenv(
            f"{config.provider_name.upper()}_DEPLOYMENT_NAME"
        )
        self.api_version = config.api_version

        # Each provider gets its own traffic controller instance so concurrency is isolated
        # per provider class (avoid global bottlenecks).
        self.traffic_controller = getattr(config, "traffic_controller", None)
        if self.traffic_controller is None:
            from ..traffic_control import TrafficController

            self.traffic_controller = TrafficController(provider_name=self.provider_name)

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
    async def complete(self, prompt: str, **kwargs) -> TutorResponse:
        """Complete a prompt and return standardized response."""
        pass

    @abstractmethod
    async def stream(self, prompt: str, **kwargs) -> AsyncStreamGenerator:
        """Stream a prompt and return standardized chunks."""
        pass

    def _map_exception(self, e: Exception) -> Exception:
        """Map provider-specific exceptions to DeepTutor exceptions.
        
        Override in subclasses for provider-specific mapping.
        """
        error_str = str(e).lower()
        
        # Authentication errors
        if any(keyword in error_str for keyword in ["unauthorized", "authentication", "invalid api key", "401"]):
            return LLMAuthenticationError(str(e))
        
        # Rate limit errors
        if any(keyword in error_str for keyword in ["rate limit", "429", "quota exceeded", "too many requests"]):
            return LLMRateLimitError(str(e))
        
        # Context window errors
        if any(keyword in error_str for keyword in ["context window", "too long", "maximum length", "token limit"]):
            return ProviderContextWindowError(str(e))
        
        # Timeout errors
        if any(keyword in error_str for keyword in ["timeout", "connection timeout", "read timeout", "408"]):
            return LLMTimeoutError(str(e))
        
        # API errors
        if any(keyword in error_str for keyword in ["500", "502", "503", "504", "internal server error"]):
            return LLMAPIError(str(e))
        
        # Configuration errors
        if any(keyword in error_str for keyword in ["configuration", "invalid configuration", "missing"]):
            return LLMConfigurationError(str(e))
        
        # Default to base error
        return LLMBaseError(str(e))

    def calculate_cost(self, usage: Dict[str, Any]) -> float:
        """Calculate cost (can be overridden)."""
        # Default implementation
        return 0.0

    def validate_config(self):
        """Validate provider-specific configuration."""
        pass  # Override in subclasses

    async def execute_with_retry(self, func: Callable, *args, max_retries=3, **kwargs):
        """Executes a function with exponential backoff retry logic under the traffic controller.
        
        Maps provider exceptions to DeepTutor exceptions.
        """

        for attempt in range(max_retries + 1):
            try:
                async with self.traffic_controller:
                    return await func(*args, **kwargs)
            except Exception as e:
                # Map to our exception hierarchy
                mapped_e = self._map_exception(e)
                
                # Check if retriable
                is_retriable = isinstance(
                    mapped_e, 
                    (LLMRateLimitError, LLMAPIError, LLMTimeoutError)
                )

                if attempt >= max_retries or not is_retriable:
                    raise mapped_e

                delay = (1.5**attempt) + (random.random() * 0.5)
                logger.warning(
                    "%s call failed (%s). Retrying in %.2fs...",
                    self.provider_name,
                    mapped_e,
                    delay,
                )
                await asyncio.sleep(delay)
