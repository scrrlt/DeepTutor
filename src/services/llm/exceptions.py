"""
Unified Exception Handling for LLM Providers.
"""


class LLMBaseError(Exception):
    """Base class for all LLM errors."""

    def __init__(self, message: str, retryable: bool = False):
        self.retryable = retryable
        super().__init__(message)


class ProviderQuotaExceededError(LLMBaseError):
    """Maps to OpenAI 429, Anthropic 429, etc."""

    def __init__(self, message: str = "Quota exceeded"):
        super().__init__(message, retryable=True)


class ProviderContextWindowError(LLMBaseError):
    """Maps to 'context_length_exceeded'"""

    def __init__(self, message: str = "Token limit reached"):
        super().__init__(message, retryable=False)


class LLMError(LLMBaseError):
    """General LLM error."""
    pass


class QuotaExceededError(LLMError):
    """Quota exceeded (retries won't help immediately)."""
    pass


class RateLimitError(LLMError):
    """Rate limit hit (retries will help)."""
    pass


class ContextWindowExceededError(LLMError):
    """Context window exceeded."""
    pass


class LLMConfigurationError(LLMError):
    """Configuration error (e.g., missing API key, invalid model)."""
    pass


class LLMProviderError(LLMError):
    """Base class for provider/API failures."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        retryable: bool = False,
    ):
        self.status_code = status_code
        super().__init__(message, retryable=retryable)


class LLMAPIError(LLMProviderError):
    """Generic API error (HTTP 4xx/5xx)."""


class LLMTimeoutError(LLMProviderError):
    """Timeout calling provider."""

    def __init__(self, message: str = "Request timed out", *, status_code: int | None = None):
        super().__init__(message, status_code=status_code, retryable=True)


class LLMRateLimitError(LLMProviderError):
    """Rate limit error (usually retryable)."""

    def __init__(self, message: str = "Rate limit exceeded", *, status_code: int | None = None):
        super().__init__(message, status_code=status_code, retryable=True)


class LLMAuthenticationError(LLMProviderError):
    """Authentication/authorization error (not retryable without config changes)."""

    def __init__(
        self,
        message: str = "Authentication failed",
        *,
        status_code: int | None = None,
    ):
        super().__init__(message, status_code=status_code, retryable=False)


def _map_error(e: Exception) -> LLMBaseError:
    """Map provider-specific errors to unified exceptions."""
    from .error_mapping import map_error

    return map_error(e)
