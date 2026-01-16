import pytest
import asyncio
from src.services.llm.error_mapping import map_error
from src.services.llm.exceptions import (
    LLMTimeoutError, 
    LLMAuthenticationError, 
    LLMRateLimitError,
    LLMAPIError
)

def test_map_timeout_error():
    """Critical: Asyncio timeouts must become LLMTimeoutError for retries."""
    exc = asyncio.TimeoutError("Too slow")
    mapped = map_error(exc)
    assert isinstance(mapped, LLMTimeoutError)

def test_map_sdk_auth_error():
    """SDK specific errors should map correctly."""
    class AuthError(Exception):
        status_code = 401

    exc = AuthError("Bad Key")
    mapped = map_error(exc)
    assert isinstance(mapped, LLMAuthenticationError)

def test_map_heuristic_error():
    """Fallback text matching."""
    exc = Exception("Error 429: Too Many Requests")
    mapped = map_error(exc)
    assert isinstance(mapped, LLMRateLimitError)

def test_do_not_swallow_runtime_errors():
    """
    CRITICAL: Internal bugs (KeyError, TypeError) must NOT be wrapped
    as API errors. They should bubble up to crash/alert the app.
    """
    bug = KeyError("missing_field")
    mapped = map_error(bug)
    
    # Should remain a KeyError, not an LLMAPIError
    assert isinstance(mapped, KeyError)
    assert not isinstance(mapped, LLMAPIError)

def test_double_mapping_safety():
    """If an error is already an LLMError, don't wrap it again."""
    original = LLMRateLimitError("Already mapped")
    mapped = map_error(original)
    assert mapped is original