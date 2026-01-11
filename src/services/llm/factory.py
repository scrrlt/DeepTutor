import os

from .config import LLMConfig
from .registry import get_provider_class, list_providers

# Import providers package to trigger registration of all provider classes

class LLMFactory:
    """
    Factory class to create LLM provider instances.
    Refactored to use Registry Pattern (no if/else chains).
    
    This replaces the old monolithic factory that had hardcoded logic for 
    cloud vs. local providers. Now, everything is a 'Provider' in the registry.
    """
    
    @staticmethod
    def create(provider_name: str, **kwargs) -> Any:
        """
        Create an instance of an LLM provider.
        
        Args:
            provider_name: The name of the provider (e.g., 'openai', 'ollama', 'azure')
            **kwargs: Additional configuration overrides
            
        Returns:
            An instance of BaseLLMProvider
        """
        # 1. Create Config Object
        # The config object now handles loading env vars specific to the provider
        config = LLMConfig(provider_name, **kwargs)
        
        # 2. Lookup Provider Class from Registry
        provider_class = get_provider_class(provider_name)
        
        if not provider_class:
            # Fallback logic: check if it's a known 'openai_compatible' flavor
            # This allows users to say provider="deepseek" and auto-map it
            from .providers.openai_compatible import KNOWN_ENDPOINTS
            if provider_name.lower() in KNOWN_ENDPOINTS:
                 # It's a flavor of OpenAI compatible (like deepseek), so use that provider
                 # and pass the flavor name as config
                 provider_class = get_provider_class("openai_compatible")
                 kwargs['flavor'] = provider_name
                 config = LLMConfig("openai_compatible", **kwargs)
            else:
                available = ", ".join(list_providers())
                raise ValueError(f"Unknown provider '{provider_name}'. Available: {available}")
            
        # 3. Instantiate the Provider
        return provider_class(config)

    @staticmethod
    def create_from_env() -> Any:
        """
        Creates an LLM provider instance based on the LLM_PROVIDER environment variable.

        If LLM_PROVIDER is not set, defaults to 'openai'.

        Returns:
            An instance of BaseLLMProvider
        Creates provider based on LLM_PROVIDER env var.
        Defaults to 'openai' if not set.
        """
        # Get the provider name from the environment variable
        # or default to 'openai' if not set
        provider_name = os.getenv("LLM_PROVIDER", "openai")

        # Create the provider instance
        return LLMFactory.create(provider_name)
