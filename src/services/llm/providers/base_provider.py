# -*- coding: utf-8 -*-
"""
Base LLM Provider - Unified interface and configuration.
"""

from abc import ABC, abstractmethod
import logging
from typing import Any, Callable, Coroutine, Dict, TypeVar
import warnings

from ..traffic_control import TrafficController

T = TypeVar("T")

from ....utils.error_rate_tracker import record_provider_call
from ....utils.network.circuit_breaker import (
    is_call_allowed,
    record_call_failure,
    record_call_success,
)
from ..error_mapping import map_error
from ..exceptions import (
    LLMError,
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
            self.traffic_controller = TrafficController(provider_name=self.provider_name)

    def _check_deprecated_kwargs(self, kwargs: dict[str, Any]) -> None:
        """Check for and warn about deprecated parameters."""
        if "max_retries" in kwargs:
            warnings.warn(
                "The 'max_retries' parameter is deprecated and ignored in the provider. "
                "Retries are now handled by the factory/tenacity.",
                DeprecationWarning,
                stacklevel=3,
            )
            kwargs.pop("max_retries")

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

    async def execute_guarded(
        self,
        func: Callable[..., Coroutine[Any, Any, T]],
        *args,
        **kwargs,
    ) -> T:
        """
        Execute provider call with circuit breaker and traffic control.
        Renamed from 'execute_with_retry' since retry logic is external.

        Args:
            func: Coroutine function to execute.
            *args: Positional arguments for func.
            **kwargs: Keyword arguments for func.

        Returns:
            Result of the wrapped call.

        Raises:
            LLMError: If the circuit breaker is open or the call fails.
        """
        self._check_deprecated_kwargs(kwargs)
        # 1. Circuit Breaker Check
        if not is_call_allowed(self.provider_name):
            record_provider_call(self.provider_name, success=False)
            raise LLMError(f"Circuit breaker open for provider {self.provider_name}")

        try:
            # 2. Traffic Control (Semaphore/Rate Limiter)
            async with self.traffic_controller:
                result = await func(*args, **kwargs)

                # 3. Success Telemetry
                record_provider_call(self.provider_name, success=True)
                record_call_success(self.provider_name)
                return result

        except Exception as e:
            # 4. Error Mapping & Telemetry
            mapped_e = self._map_exception(e)
            record_provider_call(self.provider_name, success=False)
            # FIX: Properly record failure for circuit breaker
            record_call_failure(self.provider_name)

            # Raise mapped exception up to Factory for retry decision
            raise mapped_e from e
