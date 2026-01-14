"""
Base LLM Provider - Unified interface and configuration.
"""

from abc import ABC, abstractmethod
import os
from typing import Any, Dict

from src.utils.error_rate_tracker import ErrorRateTracker
from src.utils.error_utils import format_exception_message


class BaseLLMProvider(ABC):
    """Base class for all LLM providers with unified config."""

    def __init__(self, provider_name: str):
        """Initialize with unified config fetching."""
        self.provider_name = provider_name
        self.error_rate_tracker = ErrorRateTracker()
        self.api_key = os.getenv(f"{provider_name.upper()}_API_KEY")
        self.base_url = os.getenv(f"{provider_name.upper()}_BASE_URL", self._default_base_url())
        self.deployment = os.getenv(f"{provider_name.upper()}_DEPLOYMENT_NAME")
        self.api_version = os.getenv(f"{provider_name.upper()}_API_VERSION")

        if not self.api_key:
            raise ValueError(f"Missing env var: {provider_name.upper()}_API_KEY")

    def _default_base_url(self) -> str:
        """Default base URL for the provider."""
        return ""

    @property
    def extra_headers(self) -> Dict[str, str]:
        """Provider-specific headers."""
        if self.api_version:
            return {"api-key": self.api_key}  # Azure style
        return {"Authorization": f"Bearer {self.api_key}"}  # OpenAI style

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

    def record_success(self) -> None:
        self.error_rate_tracker.record(True)

    def record_failure(self, exc: Exception) -> str:
        self.error_rate_tracker.record(False)
        return format_exception_message(exc)
