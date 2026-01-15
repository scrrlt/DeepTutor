"""
Base LLM Provider - Unified interface and configuration.
"""

from abc import ABC, abstractmethod
import asyncio
import logging
import random
from typing import Any, Awaitable, Callable, Dict

from ...utils.error_rate_tracker import record_provider_call
from ...utils.network.circuit_breaker import (
    is_call_allowed,
    record_call_failure,
    record_call_success,
)
from ..config import LLMConfig
from ..error_mapping import map_error
from ..exceptions import (
    LLMAPIError,
    LLMError,
    LLMRateLimitError,
    LLMTimeoutError,
)
from ..types import AsyncStreamGenerator, TutorResponse

logger = logging.getLogger(__name__)

# Cap retry delays to avoid excessive waits during outages.
MAX_RETRY_DELAY_SECONDS = 60.0


class BaseLLMProvider(ABC):
    """Base class for all LLM providers with unified config and retries."""

    def __init__(self, config: LLMConfig) -> None:
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
    async def complete(self, prompt: str, **kwargs: Any) -> TutorResponse:
        pass

    @abstractmethod
    async def stream(self, prompt: str, **kwargs: Any) -> AsyncStreamGenerator:
        pass

    def _map_exception(self, e: Exception) -> LLMError:
        return map_error(e, provider=self.provider_name)

    def calculate_cost(self, usage: Dict[str, Any]) -> float:
        """Placeholder for cost calculation logic."""
        return 0.0

    def _check_circuit_breaker(self) -> None:
        """Raise when the circuit breaker is open for this provider."""
        if not is_call_allowed(self.provider_name):
            record_provider_call(self.provider_name, success=False)
            error = LLMError(
                f"Circuit breaker open for provider {self.provider_name}"
            )
            setattr(error, "is_circuit_breaker", True)
            raise error

    async def execute(
        self,
        func: Callable[..., Awaitable[Any]],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """
        Execute function with circuit breaker and traffic control.

        This method does not perform retries; use execute_with_retry for
        retry/backoff behavior when establishing connections.
        """
        self._check_circuit_breaker()

        try:
            async with self.traffic_controller:
                result = await func(*args, **kwargs)
                record_provider_call(self.provider_name, success=True)
                record_call_success(self.provider_name)
                return result
        except Exception as e:
            record_provider_call(self.provider_name, success=False)
            record_call_failure(self.provider_name)
            mapped_e = self._map_exception(e)
            raise mapped_e from e

    async def execute_with_retry(
        self,
        func: Callable[..., Awaitable[Any]],
        *args: Any,
        max_retries: int = 3,
        **kwargs: Any,
    ) -> Any:
        """Retry wrapper with exponential backoff for transient errors."""
        self._check_circuit_breaker()

        for attempt in range(max_retries + 1):
            try:
                async with self.traffic_controller:
                    result = await func(*args, **kwargs)
                    record_provider_call(self.provider_name, success=True)
                    record_call_success(self.provider_name)
                    return result
            except Exception as e:
                mapped_e = self._map_exception(e)

                is_retriable = isinstance(
                    mapped_e,
                    (LLMRateLimitError, LLMTimeoutError),
                )
                if isinstance(mapped_e, LLMAPIError):
                    status_code = getattr(mapped_e, "status_code", None)
                    if status_code is not None and status_code >= 500:
                        is_retriable = True

                if attempt >= max_retries or not is_retriable:
                    record_provider_call(self.provider_name, success=False)
                    record_call_failure(self.provider_name)
                    raise mapped_e from e

                delay = (1.5**attempt) + (random.random() * 0.5)
                if isinstance(mapped_e, LLMRateLimitError):
                    retry_after = getattr(mapped_e, "retry_after", None)
                    try:
                        retry_after_value = float(retry_after)
                    except (TypeError, ValueError):
                        retry_after_value = None
                    if retry_after_value is not None:
                        delay = max(
                            0.0,
                            min(retry_after_value, MAX_RETRY_DELAY_SECONDS),
                        )
                logger.warning(
                    "[%s] Call failed. Retry %d/%d in %.2fs. Error: %s",
                    self.provider_name,
                    attempt + 1,
                    max_retries,
                    delay,
                    str(mapped_e),
                )
                await asyncio.sleep(delay)
