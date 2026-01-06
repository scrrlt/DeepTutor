"""
LLM Service
===========

Unified LLM client for all DeepTutor modules.

Usage:
    from src.services.llm import get_llm_client, LLMClient, LLMConfig

    # Get singleton client
    client = get_llm_client()
    response = await client.complete("Hello, world!")

    # Or create custom client
    config = LLMConfig(model="gpt-4o", api_key="...", base_url="...")
    client = LLMClient(config)
    
    # Provider management
    from src.services.llm import provider_manager, LLMProvider
    providers = provider_manager.list_providers()
"""

from .client import LLMClient, get_llm_client, reset_llm_client
from .config import (
    LLMConfig,
    get_llm_config,
    uses_max_completion_tokens,
    get_token_limit_kwargs,
)
from .provider import (
    LLMProvider,
    LLMProviderManager,
    provider_manager,
)

__all__ = [
    # Client
    "LLMClient",
    "get_llm_client",
    "reset_llm_client",
    # Config
    "LLMConfig",
    "get_llm_config",
    "uses_max_completion_tokens",
    "get_token_limit_kwargs",
    # Provider
    "LLMProvider",
    "LLMProviderManager",
    "provider_manager",
]
