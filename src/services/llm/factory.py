import logging
from typing import Dict, Tuple, Type

from .exceptions import LLMConfigurationError
from .providers.anthropic import AnthropicProvider
from .providers.azure import AzureProvider
from .providers.base_provider import BaseLLMProvider
from .providers.configs import (
    AnthropicConfig,
    AzureConfig,
    BaseProviderConfig,
    DeepSeekConfig,
    GeminiConfig,
    OpenAIConfig,
)
from .providers.gemini import GeminiProvider
from .providers.openai import OpenAIProvider
from .providers.openai_compatible import OpenAICompatibleProvider
from .providers.ollama import OllamaProvider

logger = logging.getLogger(__name__)


SUPPORTED_PROVIDERS: Dict[str, Tuple[Type[BaseLLMProvider], Type[BaseProviderConfig]]] = {
    "openai": (OpenAIProvider, OpenAIConfig),
    "openai_compatible": (OpenAICompatibleProvider, OpenAIConfig),
    "azure": (AzureProvider, AzureConfig),
    "gemini": (GeminiProvider, GeminiConfig),
    "anthropic": (AnthropicProvider, AnthropicConfig),
    "deepseek": (OpenAICompatibleProvider, DeepSeekConfig),
    "ollama": (OllamaProvider, BaseProviderConfig),
}


class LLMFactory:
    """Factory for creating LLM provider instances."""

    @classmethod
    def create_provider(cls, provider_type: str, model: str, **kwargs) -> BaseLLMProvider:
        """
        Create a provider instance based on type and validated config.

        Args:
            provider_type: Type of provider (openai, azure, etc.)
            model: The specific model string
            **kwargs: Config overrides (api_key, base_url, etc.)
        """
        ptype = provider_type.lower()
        if ptype not in SUPPORTED_PROVIDERS:
            available = ", ".join(sorted(SUPPORTED_PROVIDERS.keys()))
            raise LLMConfigurationError(
                f"Unsupported LLM provider type: '{provider_type}'. Available: {available}"
            )

        provider_class, config_class = SUPPORTED_PROVIDERS[ptype]
        config_obj = config_class(model=model, **kwargs)

        return provider_class(config=config_obj)

    @classmethod
    def create_from_env(cls) -> BaseLLMProvider:
        """
        Convenience method to create the default provider from environment variables.
        Used by legacy code and simple agent initializations.
        """
        import os

        ptype = os.getenv("LLM_PROVIDER", "openai").lower()
        model = os.getenv("LLM_MODEL")

        if not model:
            raise LLMConfigurationError("LLM_MODEL environment variable must be set")

        return cls.create_provider(ptype, model)


# Global Instance
llm_factory = LLMFactory()
