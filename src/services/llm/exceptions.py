"""LLM service exceptions.

Custom exception hierarchy for the LLM service.
"""

from __future__ import annotations

from typing import Any


class LLMError(Exception):
    """Base exception for all LLM-related errors."""

    def __init__(
        self,
        message: str,
        details: dict[str, Any] | None = None,
        provider: str | None = None,
        request_id: str | None = None,
    ):
        """Initialize an LLMError with optional details and provider."""
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.provider = provider
        self.request_id = request_id

    @property
    def is_retryable(self) -> bool:
        """Whether the caller should retry this error."""
        return False

    def __str__(self) -> str:
        """Return a formatted error string with provider and details when set."""
        parts: list[str] = []
        if self.provider:
            parts.append(f"[{self.provider}]")
        parts.append(self.message)
        if self.request_id:
            parts.append(f"(req_id: {self.request_id})")
        if self.details:
            parts.append(f"(details: {self.details})")
        return " ".join(parts)


class LLMConfigError(LLMError):
    """Raised when there's an error in LLM configuration."""

    pass


class LLMProviderError(LLMError):
    """Raised when there's an error with the LLM provider."""

    pass


class LLMAPIError(LLMError):
    """
    Raised when an API call to an LLM provider fails.

    Standardizes status_code and provider name.
    """

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        provider: str | None = None,
        details: dict[str, Any] | None = None,
        request_id: str | None = None,
    ):
        """Initialize API error with status code and provider context."""
        super().__init__(message, details, provider, request_id)
        self.status_code = status_code

    @property
    def is_retryable(self) -> bool:
        if self.status_code is None:
            return False
        return self.status_code >= 500

    def __str__(self) -> str:
        """Return formatted string including provider and status code."""
        base_str = super().__str__()
        if self.status_code is not None:
            return f"HTTP {self.status_code} {base_str}"
        return base_str


class LLMTimeoutError(LLMAPIError):
    """Raised when an API call times out."""

    def __init__(
        self,
        message: str = "Request timed out",
        timeout: float | None = None,
        provider: str | None = None,
        request_id: str | None = None,
    ):
        """Initialize timeout error with optional timeout value."""
        super().__init__(
            message,
            status_code=408,
            provider=provider,
            request_id=request_id,
        )
        self.timeout = timeout

    @property
    def is_retryable(self) -> bool:
        return True


class LLMRateLimitError(LLMAPIError):
    """Raised when rate limited by the API."""

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        retry_after: float | None = None,
        provider: str | None = None,
        details: dict[str, Any] | None = None,
        request_id: str | None = None,
    ):
        """Initialize rate limit error with optional retry_after value."""
        super().__init__(
            message,
            status_code=429,
            provider=provider,
            details=details,
            request_id=request_id,
        )
        self.retry_after = retry_after

    @property
    def is_retryable(self) -> bool:
        return True


class LLMServiceUnavailableError(LLMAPIError):
    """Raised when the service is overloaded or down (HTTP 503)."""

    def __init__(
        self,
        message: str = "Service unavailable",
        provider: str | None = None,
        details: dict[str, Any] | None = None,
        request_id: str | None = None,
    ):
        super().__init__(
            message,
            status_code=503,
            provider=provider,
            details=details,
            request_id=request_id,
        )

    @property
    def is_retryable(self) -> bool:
        return True


class LLMAuthenticationError(LLMAPIError):
    """Raised when authentication fails (invalid API key, etc.)."""

    def __init__(
        self,
        message: str = "Authentication failed",
        provider: str | None = None,
        details: dict[str, Any] | None = None,
        request_id: str | None = None,
    ):
        """Initialize authentication error."""
        super().__init__(
            message,
            status_code=401,
            provider=provider,
            details=details,
            request_id=request_id,
        )

    @property
    def is_retryable(self) -> bool:
        return False


class LLMModelNotFoundError(LLMAPIError):
    """Raised when the requested model is not found."""

    def __init__(
        self,
        message: str = "Model not found",
        model: str | None = None,
        provider: str | None = None,
    ):
        """Initialize model-not-found error with optional model name."""
        super().__init__(message, status_code=404, provider=provider)
        self.model = model


class LLMParseError(LLMError):
    """Raised when parsing LLM output fails."""

    def __init__(
        self,
        message: str = "Failed to parse LLM output",
        provider: str | None = None,
        details: dict[str, Any] | None = None,
    ):
        """Initialize parse error with optional details payload."""
        super().__init__(message, details=details, provider=provider)


# Multi-provider specific aliases for mapping rules
class ProviderQuotaExceededError(LLMRateLimitError):
    """Alias for provider-specific quota exceeded errors."""


class ProviderContextWindowError(LLMAPIError):
    """Alias for provider-specific context window errors."""


class LLMCircuitBreakerError(LLMError):
    """Raised when the circuit breaker is open and calls are blocked."""

    def __init__(
        self,
        message: str = "Circuit breaker open",
        provider: str | None = None,
        details: dict[str, Any] | None = None,
        request_id: str | None = None,
    ):
        super().__init__(message, details=details, provider=provider, request_id=request_id)

    @property
    def is_retryable(self) -> bool:
        return False


__all__ = [
    "LLMError",
    "LLMConfigError",
    "LLMProviderError",
    "LLMAPIError",
    "LLMTimeoutError",
    "LLMRateLimitError",
    "LLMServiceUnavailableError",
    "LLMAuthenticationError",
    "LLMModelNotFoundError",
    "LLMParseError",
    "ProviderQuotaExceededError",
    "ProviderContextWindowError",
    "LLMCircuitBreakerError",
]
