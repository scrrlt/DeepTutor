"""
Circuit Breaker Pattern.

Prevents cascading failures by stopping requests to failing services.
"""

from collections.abc import Callable
import time
from typing import Any, TypeVar

R = TypeVar("R")


class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open."""


class CircuitBreaker:
    """Circuit breaker implementation to prevent cascading failures."""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 30,
    ) -> None:
        """Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening circuit.
            recovery_timeout: Time in seconds to wait before half-open state.
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failures = 0
        self.last_failure_time = 0.0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF-OPEN

    async def call(
        self,
        func: Callable[..., Any],
        *args: Any,  # noqa: ANN401
        **kwargs: Any,  # noqa: ANN401
    ) -> Any:  # noqa: ANN401
        """Execute function with circuit breaker protection.

        Args:
            func: The async function to call.
            *args: Positional arguments for the function.
            **kwargs: Keyword arguments for the function.

        Returns:
            The result of the function call.

        Raises:
            CircuitBreakerOpenError: If circuit is open.
        """
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF-OPEN"
            else:
                msg = "Circuit is OPEN"
                raise CircuitBreakerOpenError(msg)

        try:
            result = await func(*args, **kwargs)
            if self.state == "HALF-OPEN":
                self.reset()
            return result
        except Exception:
            self.failures += 1
            self.last_failure_time = time.time()
            if self.failures >= self.failure_threshold:
                self.state = "OPEN"
            raise

    def reset(self) -> None:
        """Reset circuit breaker to closed state."""
        self.failures = 0
        self.state = "CLOSED"


# Global instance for backward compatibility
_circuit_breaker = CircuitBreaker()


def is_call_allowed(provider: str) -> bool:  # noqa: ARG001
    """Check if call is allowed by circuit breaker (synchronous wrapper).

    Args:
        provider: Provider name (unused in current implementation).

    Returns:
        True if call is allowed, False otherwise.
    """
    # For simplicity, since the new implementation is per-instance, not per-provider,
    # we'll use a global instance. In a real implementation, you'd have per-provider instances.
    try:
        # Since this is called synchronously, we can't await
        # For now, just return True if not in OPEN state
        return _circuit_breaker.state != "OPEN"
    except Exception:  # noqa: BLE001
        return True


def record_call_success(provider: str) -> None:  # noqa: ARG001
    """Record successful call.

    Args:
        provider: Provider name (unused in current implementation).
    """
    if _circuit_breaker.state == "HALF-OPEN":
        _circuit_breaker.reset()


def record_call_failure(provider: str) -> None:  # noqa: ARG001
    """Record failed call.

    Args:
        provider: Provider name (unused in current implementation).
    """
    _circuit_breaker.failures += 1


def alert_callback(provider: str, rate: float) -> None:
    """Default alert callback for error rate tracking.

    Args:
        provider: Provider name triggering the alert.
        rate: Error rate that exceeded the threshold.
    Returns:
        None.
    Raises:
        None.
    """
    _ = (provider, rate)
    _circuit_breaker.last_failure_time = time.time()
    if _circuit_breaker.failures >= _circuit_breaker.failure_threshold:
        _circuit_breaker.state = "OPEN"
