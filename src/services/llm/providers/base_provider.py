"""
Base LLM Provider - Unified interface and configuration.
"""

from abc import ABC, abstractmethod
import asyncio
import logging
import random
from typing import Any, Callable, Dict

from ..error_mapping import map_error
from ..exceptions import (
    LLMAPIError,
    LLMError,
    LLMRateLimitError,
    LLMTimeoutError,
)
from ..types import AsyncStreamGenerator, TutorResponse

logger = logging.getLogger(__name__)


class BaseLLMProvider(ABC):
    """Base class for all LLM providers with unified config and retries."""

    def __init__(self, config):
        self.config = config
        self.provider_name = config.provider_name
        self.api_key = config.api_key
        self.base_url = getattr(config, "base_url", "")

        # Isolation: Each provider gets its own traffic controller instance
        self.traffic_controller = getattr(config, "traffic_controller", None)
        if self.traffic_controller is None:
            from ..traffic_control import TrafficController

            self.traffic_controller = TrafficController(provider_name=self.provider_name)

    @abstractmethod
    async def complete(self, prompt: str, **kwargs) -> TutorResponse:
        pass

    @abstractmethod
    async def stream(self, prompt: str, **kwargs) -> AsyncStreamGenerator:
        pass

    def _map_exception(self, e: Exception) -> LLMError:
        return map_error(e, provider=self.provider_name)

    def calculate_cost(self, usage: Dict[str, Any]) -> float:
        """Placeholder for cost calculation logic."""
        return 0.0

    async def execute_with_retry(self, func: Callable, *args, max_retries=3, **kwargs):
        """Standard retry wrapper with exponential backoff."""
        for attempt in range(max_retries + 1):
            try:
                async with self.traffic_controller:
                    return await func(*args, **kwargs)
            except Exception as e:
                mapped_e = self._map_exception(e)

                # Logic check: should we retry?
                is_retriable = isinstance(mapped_e, (LLMRateLimitError, LLMTimeoutError))
                if isinstance(mapped_e, LLMAPIError):
                    status_code = getattr(mapped_e, "status_code", None)
                    if status_code is not None and status_code >= 500:
                        is_retriable = True

                if attempt >= max_retries or not is_retriable:
                    raise mapped_e

                delay = (1.5**attempt) + (random.random() * 0.5)
                logger.warning(
                    "[%s] Call failed. Retry %d/%d in %.2fs. Error: %s",
                    self.provider_name,
                    attempt + 1,
                    max_retries,
                    delay,
                    str(mapped_e),
                )
                await asyncio.sleep(delay)
