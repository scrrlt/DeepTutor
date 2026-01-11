"""
LLM Provider Wrappers - Middleware pattern for adding features.
"""

from typing import Any
from .providers.base_provider import BaseLLMProvider
from .utils.circuit_breaker import CircuitBreaker


class ResilientProviderWrapper(BaseLLMProvider):
    """
    Wraps ANY provider (OpenAI, Anthropic, Ollama) and adds
    Circuit Breaker and Retry logic automatically.
    """

    def __init__(self, provider: BaseLLMProvider):
        # Don't call super().__init__ since we're wrapping
        self.provider = provider
        self.breaker = CircuitBreaker(failure_threshold=5, timeout=60)

    async def complete(self, prompt: str, **kwargs) -> str:
        """
        Intercepts the call, runs it through circuit breaker,
        then delegates to the actual provider.
        """
        return await self.breaker.call_async(
            self.provider.complete,
            prompt,
            **kwargs
        )

    # Delegate attribute access to the underlying provider
    def __getattr__(self, name):
        return getattr(self.provider, name)
