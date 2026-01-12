from __future__ import annotations

from enum import Enum
from typing import Any, AsyncGenerator, Dict, List, Optional
import os

from .config import LLMConfig
from .registry import get_provider_class, list_providers

# Import providers package to trigger registration of all provider classes
import src.services.llm.providers


DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 1.0
DEFAULT_EXPONENTIAL_BACKOFF = True


API_PROVIDER_PRESETS: Dict[str, Dict[str, Any]] = {}
LOCAL_PROVIDER_PRESETS: Dict[str, Dict[str, Any]] = {}


class LLMMode(str, Enum):
    default = "default"


def get_llm_mode() -> LLMMode:
    return LLMMode.default


def get_mode_info() -> Dict[str, Any]:
    return {"mode": get_llm_mode().value}


def get_provider_presets() -> Dict[str, Dict[str, Any]]:
    return {"api": API_PROVIDER_PRESETS, "local": LOCAL_PROVIDER_PRESETS}


async def fetch_models(*args, **kwargs) -> List[str]:
    binding = kwargs.get("binding")
    base_url = kwargs.get("base_url")
    api_key = kwargs.get("api_key")

    if not base_url:
        return []

    from .utils import is_local_llm_server, sanitize_url

    sanitized = sanitize_url(base_url)
    if is_local_llm_server(sanitized):
        from . import local_provider

        return await local_provider.fetch_models(base_url=sanitized, api_key=api_key)

    from . import cloud_provider

    return await cloud_provider.fetch_models(
        base_url=sanitized,
        api_key=api_key,
        binding=binding or "openai",
    )


async def complete(*args, **kwargs) -> str:
    # Support both positional and keyword invocation.
    prompt = kwargs.get("prompt")
    if prompt is None and args:
        prompt = args[0]

    system_prompt = kwargs.get("system_prompt", "You are a helpful assistant.")
    model = kwargs.get("model")
    api_key = kwargs.get("api_key")
    base_url = kwargs.get("base_url")
    binding = kwargs.get("binding")
    api_version = kwargs.get("api_version")
    messages = kwargs.get("messages")

    from .utils import is_local_llm_server, sanitize_url

    if base_url:
        base_url = sanitize_url(base_url, model or "")

    # If base_url looks local, route to local provider.
    if base_url and is_local_llm_server(base_url):
        from . import local_provider

        return await local_provider.complete(
            prompt=prompt or "",
            system_prompt=system_prompt,
            model=model,
            api_key=api_key,
            base_url=base_url,
            messages=messages,
            **{k: v for k, v in kwargs.items() if k not in {"prompt", "system_prompt", "model", "api_key", "base_url", "binding", "api_version", "messages"}},
        )

    from . import cloud_provider

    return await cloud_provider.complete(
        prompt=prompt or "",
        system_prompt=system_prompt,
        model=model,
        api_key=api_key,
        base_url=base_url,
        binding=binding or "openai",
        api_version=api_version,
        messages=messages,
        **{k: v for k, v in kwargs.items() if k not in {"prompt", "system_prompt", "model", "api_key", "base_url", "binding", "api_version", "messages"}},
    )


async def stream(*args, **kwargs) -> AsyncGenerator[str, None]:
    # Support both positional and keyword invocation.
    prompt = kwargs.get("prompt")
    if prompt is None and args:
        prompt = args[0]

    system_prompt = kwargs.get("system_prompt", "You are a helpful assistant.")
    model = kwargs.get("model")
    api_key = kwargs.get("api_key")
    base_url = kwargs.get("base_url")
    binding = kwargs.get("binding")
    api_version = kwargs.get("api_version")
    messages = kwargs.get("messages")

    from .utils import is_local_llm_server, sanitize_url

    if base_url:
        base_url = sanitize_url(base_url, model or "")

    if base_url and is_local_llm_server(base_url):
        from . import local_provider

        async for chunk in local_provider.stream(
            prompt=prompt or "",
            system_prompt=system_prompt,
            model=model,
            api_key=api_key,
            base_url=base_url,
            messages=messages,
            **{k: v for k, v in kwargs.items() if k not in {"prompt", "system_prompt", "model", "api_key", "base_url", "binding", "api_version", "messages"}},
        ):
            yield chunk
        return

    from . import cloud_provider

    async for chunk in cloud_provider.stream(
        prompt=prompt or "",
        system_prompt=system_prompt,
        model=model,
        api_key=api_key,
        base_url=base_url,
        binding=binding or "openai",
        api_version=api_version,
        messages=messages,
        **{k: v for k, v in kwargs.items() if k not in {"prompt", "system_prompt", "model", "api_key", "base_url", "binding", "api_version", "messages"}},
    ):
        yield chunk

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
        """
        # Get the provider name from the environment variable
        # or default to 'openai' if not set
        provider_name = os.getenv("LLM_PROVIDER", "openai")

        # Create the provider instance
        return LLMFactory.create(provider_name)
