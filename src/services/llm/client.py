"""LLM client interface.

Stable wrapper around the LLM factory functions for agent and service access.
Provides convenience methods for LLM completion with configuration management.
"""

from collections.abc import Awaitable, Callable
from typing import Any

from src.logging import get_logger

from .exceptions import LLMClientError
from .config import LLMConfig, get_llm_config


class LLMClient:
    """
    Unified LLM client for all services.

    Provides a clean, stable interface for agents to interact with LLMs.
    Wraps providers and handles metrics, specialized logging, and compatibility adapters.
    """

    def __init__(self, config: LLMConfig | None = None) -> None:
        """Initialize LLM client."""
        self.config = config or get_llm_config()
        self.logger = get_logger("LLMClient")
        self._provider = None

    @property
    def provider(self):
        """Lazy-loaded provider instance."""
        if self._provider is None:
            from .factory import LLMFactory

    def _setup_openai_env_vars(self):
        """
        Set OpenAI environment variables for LightRAG compatibility.

        LightRAG's internal functions read from os.environ["OPENAI_API_KEY"]
        even when api_key is passed as parameter. This method ensures the
        environment variables are set for all LightRAG operations.
        """
        import os

        binding = getattr(self.config, "binding", "openai")
        binding_lower = binding.lower() if isinstance(binding, str) else "openai"

        # Only set env vars for OpenAI-compatible bindings
        if binding_lower in ("openai", "azure", "azure_openai", "gemini"):
            if self.config.api_key:
                os.environ["OPENAI_API_KEY"] = self.config.api_key
                self.logger.debug("Set OPENAI_API_KEY env var for LightRAG compatibility")

            if self.config.base_url:
                os.environ["OPENAI_BASE_URL"] = self.config.base_url
                self.logger.debug("Set OPENAI_BASE_URL env var to %s", self.config.base_url)

    async def complete(
        self,
        prompt: str,
        system_prompt: str | None = None,
        history: list[dict[str, str]] | None = None,
        **kwargs: Any,
    ) -> str:
        """Call LLM completion.

        Args:
            prompt: User prompt
            system_prompt: System instruction
            history: Conversation history
            **kwargs: Additional arguments

        Returns:
            Completion text
        """
        try:
            return await self.provider.complete(
                prompt=prompt, system_prompt=system_prompt, history=history, **kwargs
            )
        except Exception as e:
            self.logger.error(f"LLM failure: {e}")
            raise LLMClientError(f"Completion failed: {e}") from e

    async def stream(
        self,
        prompt: str,
        system_prompt: str | None = None,
        history: list[dict[str, str]] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Stream LLM completion."""
        try:
            return await self.provider.stream(
                prompt=prompt, system_prompt=system_prompt, history=history, **kwargs
            )
        except Exception as e:
            self.logger.error(f"LLM streaming failure: {e}")
            raise LLMClientError(f"Streaming failed: {e}") from e

    def get_model_func(self) -> Callable[..., Awaitable[str]]:
        """
        Get a function compatible with LightRAG's llm_model_func parameter.
        """

        async def llm_model_func(
            prompt: str,
            system_prompt: str | None = None,
            history_messages: list[dict[str, Any]] | None = None,
            **kwargs: Any,
        ) -> str:
            return await self.complete(
                prompt=prompt, system_prompt=system_prompt, history=history_messages, **kwargs
            )

        return llm_model_func

    def get_vision_model_func(self) -> Callable[..., Any]:
        """
        Get a function compatible with LightRAG's vision_model_func parameter.
        """

        async def vision_model_func(
            prompt: str,
            system_prompt: str | None = None,
            history_messages: list[dict[str, Any]] | None = None,
            image_data: str | None = None,
            messages: list[dict[str, Any]] | None = None,
            **kwargs: Any,
        ) -> Any:
            return await self.complete(
                prompt=prompt,
                system_prompt=system_prompt,
                history=history_messages,
                image_data=image_data,
                messages=messages,
                **kwargs,
            )

        return vision_model_func


_client: LLMClient | None = None


def get_llm_client(config: LLMConfig | None = None) -> LLMClient:
    """
    Get or create the singleton LLM client.

    Args:
        config: Optional configuration override. If client exists, this is ignored
                (with a warning).

    Returns:
        The singleton LLMClient instance.
    """

    global _client
    if _client is None:
        _client = LLMClient(config)
    elif config is not None:
        import warnings

        warnings.warn(
            "LLM client already initialized; provided config will be ignored. "
            "Call reset_llm_client() first to use a different config.",
            stacklevel=2,
        )
    return _client


def reset_llm_client() -> None:
    """Reset the singleton LLM client."""

    global _client
    _client = None
