from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Callable, AsyncGenerator
import os
import asyncio
import random
import logging

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
                return await func(*args, **kwargs)
            except Exception as e:
                error_str = str(e).lower()
                is_retriable = (
                    "429" in error_str or "rate limit" in error_str or
                    "quota" in error_str or "500" in error_str or
                    "503" in error_str or "timeout" in error_str
                )

                if attempt >= max_retries or not is_retriable:
                    raise e

                delay = (1.5 ** attempt) + (random.random() * 0.5)
                logger.warning(f"LLM call failed ({e}). Retrying in {delay:.2f}s...")
                await asyncio.sleep(delay)
