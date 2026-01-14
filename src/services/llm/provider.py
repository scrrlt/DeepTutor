from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Callable, AsyncGenerator
import os
import asyncio
import random
import logging
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from src.utils.error_rate_tracker import ErrorRateTracker
from src.utils.error_utils import format_exception_message

logger = logging.getLogger(__name__)

class BaseLLMProvider(ABC):
    """
    Abstract base class for all LLM providers.
    Handles common logic like cost calculation, model resolution, and retries.
    """

    price_per_input_token = 0.0
    price_per_output_token = 0.0
    provider_name = "base"

    def __init__(self, config: Any):
        self.config = config
        self.api_key = config.api_key
        self.base_url = config.base_url

        self.error_rate_tracker = ErrorRateTracker()

        if hasattr(config, "pricing"):
            self.price_per_input_token = config.pricing.get("input", self.price_per_input_token)
            self.price_per_output_token = config.pricing.get("output", self.price_per_output_token)

    @abstractmethod
    async def complete(self, prompt: str, **kwargs) -> str:
        """Generates a text completion for the given prompt."""
        pass

    async def stream(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        """Default stream implementation (yields once). Subclasses override this."""
        result = await self.complete(prompt, **kwargs)
        yield result

    def calculate_cost(self, usage: Dict[str, int]) -> float:
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)
        cost = (input_tokens * self.price_per_input_token) + \
               (output_tokens * self.price_per_output_token)
        return round(cost, 6)

    def resolve_model(self, requested_model: str) -> str:
        return requested_model

    async def execute_with_retry(self, func: Callable, *args, max_retries=3, **kwargs):
        """Executes a function with exponential backoff retry logic."""
        for attempt in range(max_retries + 1):
            try:
                result = await func(*args, **kwargs)
                self.error_rate_tracker.record(True)
                return result
            except Exception as e:
                self.error_rate_tracker.record(False)
                error_str = str(e).lower()
                is_retriable = (
                    "429" in error_str or "rate limit" in error_str or
                    "quota" in error_str or "500" in error_str or
                    "503" in error_str or "timeout" in error_str
                )

                if attempt >= max_retries or not is_retriable:
                    raise RuntimeError(format_exception_message(e)) from e

                delay = (1.5 ** attempt) + (random.random() * 0.5)
                logger.warning(f"LLM call failed ({e}). Retrying in {delay:.2f}s...")
                await asyncio.sleep(delay)


class ProviderType(str, Enum):
    api = "api"
    local = "local"


class LLMProvider(BaseModel):
    name: str
    binding: str = Field(default="openai")
    base_url: str = Field(default="")
    api_key: str = Field(default="")
    model: str = Field(default="")
    requires_key: bool = True
    provider_type: ProviderType = ProviderType.api
    is_active: bool = False


class LLMProviderManager:
    def __init__(self):
        self._providers: Dict[str, LLMProvider] = {}
        self._active: Optional[str] = None

    def list_providers(self):
        return list(self._providers.values())

    async def get_active_provider_async(self) -> Optional[LLMProvider]:
        """Async version of get_active_provider."""
        raw_providers = await self._load_providers_async()
        providers = [LLMProvider(**p) for p in raw_providers]
        for p in providers:
            if p.is_active:
                return p
        return None

    async def get_active_provider_async(self) -> Optional[LLMProvider]:
        """Async version of get_active_provider."""
        raw_providers = await self._load_providers_async()
        providers = [LLMProvider(**p) for p in raw_providers]
        for p in providers:
            if p.is_active:
                return p
        return None

    def add_provider(self, provider: LLMProvider) -> LLMProvider:
        key = provider.name
        if key in self._providers:
            raise ValueError("Provider already exists")
        self._providers[key] = provider
        if provider.is_active or self._active is None:
            self.set_active_provider(key)
        return self._providers[key]

    def update_provider(self, name: str, updates: Dict[str, Any]) -> Optional[LLMProvider]:
        if name not in self._providers:
            return None
        existing = self._providers[name]
        updated = existing.model_copy(update=updates)
        self._providers[name] = updated
        if updated.is_active:
            self.set_active_provider(name)
        return updated

    def delete_provider(self, name: str) -> bool:
        if name not in self._providers:
            return False
        was_active = self._active == name
        del self._providers[name]
        if was_active:
            self._active = None
            # pick any remaining provider
            for key in self._providers.keys():
                self.set_active_provider(key)
                break
        return True

    def set_active_provider(self, name: str) -> Optional[LLMProvider]:
        if name not in self._providers:
            return None
        self._active = name
        for key, prov in list(self._providers.items()):
            self._providers[key] = prov.model_copy(update={"is_active": key == name})
        return self._providers[name]

    def get_active_provider(self) -> Optional[LLMProvider]:
        if not self._active:
            return None
        return self._providers.get(self._active)


provider_manager = LLMProviderManager()
