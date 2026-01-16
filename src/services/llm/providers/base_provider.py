"""Base LLM provider with unified configuration and retries."""

from abc import ABC
import logging
from typing import Any, Awaitable, Callable

import tenacity
from tenacity import AsyncRetrying, retry_if_exception, stop_after_attempt

from src.utils.error_rate_tracker import record_provider_call
from src.utils.network.circuit_breaker import (
    is_call_allowed,
    record_call_failure,
    record_call_success,
)
from ..traffic_control import TrafficController
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
BASE_RETRY_DELAY_SECONDS = 1.0


class BaseLLMProvider(ABC):
    """Base class for all LLM providers with unified config and retries."""

    def __init__(self, config: LLMConfig) -> None:
        """Initialize provider with shared configuration and traffic control."""
        self.config = config
        self.provider_name = config.provider_name
        self.api_key = config.api_key
        self.base_url = getattr(config, "base_url", "")

        # Isolation: Each provider gets its own traffic controller instance
        self.traffic_controller: TrafficController
        traffic_controller = getattr(config, "traffic_controller", None)
        if isinstance(traffic_controller, TrafficController):
            self.traffic_controller = traffic_controller
        else:
            self.traffic_controller = TrafficController(
                provider_name=self.provider_name
            )

    async def complete(self, prompt: str, **kwargs: Any) -> TutorResponse:
        """Run a completion call for the provider."""
        raise NotImplementedError

    async def stream(self, prompt: str, **kwargs: Any) -> AsyncStreamGenerator:
        """Stream completion chunks for the provider."""
        raise NotImplementedError

    def _map_exception(self, e: Exception) -> LLMError:
        return map_error(e, provider=self.provider_name)

    def calculate_cost(self, usage: Dict[str, Any]) -> float:
        """Calculate cost estimate for a provider call."""
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

    def _should_record_failure(self, error: LLMError) -> bool:
        if isinstance(error, (LLMRateLimitError, LLMTimeoutError)):
            return True
        if isinstance(error, LLMAPIError):
            status_code = getattr(error, "status_code", None)
            return status_code is not None and status_code >= 500
        return False

    def _should_retry_error(self, error: BaseException) -> bool:
        if isinstance(error, (LLMRateLimitError, LLMTimeoutError)):
            return True
        if isinstance(error, LLMAPIError):
            status_code = getattr(error, "status_code", None)
            return status_code is not None and status_code >= 500
        return False

    def _wait_strategy(self, retry_state: tenacity.RetryCallState) -> float:
        exc = retry_state.outcome.exception()
        if isinstance(exc, LLMRateLimitError):
            retry_after = getattr(exc, "retry_after", None)
            retry_after_value: float | None = None
            if retry_after is not None:
                try:
                    retry_after_value = float(retry_after)
                except (TypeError, ValueError):
                    retry_after_value = None
            if retry_after_value is not None:
                return max(0.0, min(retry_after_value, MAX_RETRY_DELAY_SECONDS))

        wait_fn = tenacity.wait_exponential(
            multiplier=1.5,
            min=BASE_RETRY_DELAY_SECONDS,
            max=MAX_RETRY_DELAY_SECONDS,
        )
        return float(wait_fn(retry_state))

    async def _execute_core(
        self,
        func: Callable[..., Awaitable[Any]],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """
        Core execution pipeline:
        1) circuit breaker check
        2) traffic control context
        3) call execution
        4) mapping + metrics
        """
        self._check_circuit_breaker()

        try:
            async with self.traffic_controller:
                result = await func(*args, **kwargs)
                record_provider_call(self.provider_name, success=True)
                record_call_success(self.provider_name)
                return result
        except Exception as exc:
            mapped_exc = self._map_exception(exc)
            record_provider_call(self.provider_name, success=False)
            if self._should_record_failure(mapped_exc):
                record_call_failure(self.provider_name)
            raise mapped_exc from exc

    async def execute(
        self,
        func: Callable[..., Awaitable[Any]],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Execute a single attempt without retry."""
        return await self._execute_core(func, *args, **kwargs)

    async def execute_with_retry(
        self,
        func: Callable[..., Awaitable[Any]],
        *args: Any,
        max_retries: int = 3,
        **kwargs: Any,
    ) -> Any:
        """Execute with automatic retries using tenacity."""
        retrying = AsyncRetrying(
            stop=stop_after_attempt(max_retries + 1),
            wait=self._wait_strategy,
            retry=retry_if_exception(self._should_retry_error),
            reraise=True,
            before_sleep=tenacity.before_sleep_log(logger, logging.WARNING),
        )

        async for attempt in retrying:
            with attempt:
                return await self._execute_core(func, *args, **kwargs)
