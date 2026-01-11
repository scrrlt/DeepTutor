"""
Unified Exception Handling for LLM Providers.
"""

from .error_mapping import map_error


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


def _map_error(e: Exception) -> LLMBaseError:
    """Map provider-specific errors to unified exceptions."""
    return map_error(e)
