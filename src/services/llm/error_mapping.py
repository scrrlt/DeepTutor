"""
Error Mapping - Map provider-specific errors to unified exceptions.
"""

from .exceptions import LLMBaseError, ProviderContextWindowError, ProviderQuotaExceededError


def map_error(e: Exception) -> LLMBaseError:
    """Map provider-specific errors to unified exceptions."""
    error_str = str(e).lower()
    if "rate limit" in error_str or "429" in error_str or "quota" in error_str:
        return ProviderQuotaExceededError()
    if "maximum context length" in error_str or "context_length_exceeded" in error_str:
        return ProviderContextWindowError()
    return LLMBaseError(str(e))
