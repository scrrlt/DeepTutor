"""
Unified LLM Client - High-level interface for LLM operations.
"""

from typing import Optional
from .factory import LLMFactory
from .providers.base_provider import BaseLLMProvider


class LLMClient:
    """High-level client for LLM operations with automatic provider management."""

    def __init__(self, provider: Optional[BaseLLMProvider] = None):
        self._provider = provider

    @property
    def provider(self) -> BaseLLMProvider:
        if self._provider is None:
            self._provider = LLMFactory.create_from_env()
        return self._provider

    async def complete(self, prompt: str, **kwargs) -> str:
        """Complete a prompt using the configured provider."""
        return await self.provider.complete(prompt, **kwargs)

    async def stream(self, prompt: str, **kwargs):
        """Stream a completion using the configured provider."""
        async for chunk in self.provider.stream(prompt, **kwargs):
            yield chunk

    def calculate_cost(self, usage):
        """Calculate cost for usage."""
        return self.provider.calculate_cost(usage)


# Global client instance
_client_instance: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    """Get the global LLM client instance."""
    global _client_instance
    if _client_instance is None:
        _client_instance = LLMClient()
    return _client_instance


def reset_llm_client():
    """Reset the global client instance."""
    global _client_instance
    _client_instance = None
