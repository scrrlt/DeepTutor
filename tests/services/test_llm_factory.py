import pytest
import asyncio
from unittest.mock import AsyncMock, patch
from unittest.mock import MagicMock
# CORRECTED IMPORT PATH
from src.services.llm.factory import (
    stream, 
    complete, 
    LLMRateLimitError, 
    LLMAPIError,
    _sanitize_cache_kwargs,
    _is_retriable_llm_api_error
)
from src.services.llm.config import LLMConfig

@pytest.fixture(autouse=True)
def mock_llm_config():
    """Automatically mock LLM configuration for all tests in this file."""
    with patch("src.services.llm.factory.get_llm_config") as mock_get:
        mock_get.return_value = LLMConfig(
            model="gpt-4",
            binding="openai",
            base_url="https://api.openai.com/v1",
            api_key="sk-test-key",
        )
        yield mock_get

# Helper to create async generators for mocking
async def mock_async_gen(items, error_at_end=None):
    for item in items:
        yield item
        # Simulate network latency
        await asyncio.sleep(0.01) 
    if error_at_end:
        raise error_at_end

@pytest.mark.asyncio
async def test_stream_aborts_on_mid_stream_failure():
    """
    CRITICAL: Verifies that if a stream yields data and THEN fails, 
    we do NOT retry. Retrying here would duplicate the first chunk.
    """
    # Setup: Yields "Hello", then throws ConnectionResetError
    broken_gen = mock_async_gen(["Hello"], error_at_end=ConnectionResetError("Connection lost"))
    
    # Patch the underlying provider function used by the routing provider.
    with patch("src.services.llm.cloud_provider.stream", return_value=broken_gen):
        # We expect the exception to bubble up (no mid-stream retry).
        with pytest.raises(LLMAPIError):
            results = []
            async for chunk in stream(
                prompt="test", 
                max_retries=3, 
                retry_delay=0.01
            ):
                results.append(chunk)
        
        # Verify we got the partial data
        assert results == ["Hello"]

@pytest.mark.asyncio
async def test_stream_retries_on_initial_connection_failure():
    """
    Verifies that if the stream fails to start (Connection error),
    we retry using the configured backoff/count.
    """
    success_gen = mock_async_gen(["Success"])
    
    # Mock behavior: Fail twice (timeout), then succeed
    mock_stream_func = MagicMock()
    mock_stream_func.side_effect = [
        asyncio.TimeoutError("Timeout 1"),
        asyncio.TimeoutError("Timeout 2"),
        success_gen
    ]

    with patch("src.services.llm.cloud_provider.stream", mock_stream_func):
        results = []
        async for chunk in stream(
            prompt="test",
            max_retries=3,
            retry_delay=0.001
        ):
            results.append(chunk)

        assert results == ["Success"]
        assert mock_stream_func.call_count == 3

@pytest.mark.asyncio
@pytest.mark.parametrize("base_url, should_hit_local", [
    ("http://localhost:11434/v1", True),
    ("http://127.0.0.1:8000/v1", True),
    ("http://host.docker.internal:11434", True),
    ("https://api.openai.com/v1", False),
    ("https://my-private-cloud.com/v1", False),
])
async def test_provider_routing_logic(base_url, should_hit_local):
    """Verifies that URLs are correctly routed to Local vs Cloud providers."""
    with patch(
        "src.services.llm.local_provider.complete",
        new_callable=AsyncMock,
    ) as mock_local, patch(
        "src.services.llm.cloud_provider.complete",
        new_callable=AsyncMock,
    ) as mock_cloud:
        
        mock_local.return_value = "local_response"
        mock_cloud.return_value = "cloud_response"

        # Pass dummy API key to pass validation logic
        await complete(prompt="test", base_url=base_url, api_key="sk-xxx")

        if should_hit_local:
            mock_local.assert_called_once()
            mock_cloud.assert_not_called()
        else:
            mock_cloud.assert_called_once()
            mock_local.assert_not_called()

def test_cache_key_sanitization():
    """Ensure sensitive or transient keys don't affect cache hashing."""
    params1 = {"max_tokens": 100, "api_key": "sk-123"}
    params2 = {"max_tokens": 100, "api_key": "sk-456-rotated-key"}
    
    # These should produce identical cache dictionaries
    assert _sanitize_cache_kwargs(params1) == _sanitize_cache_kwargs(params2)
    
    # These should differ
    params3 = {"max_tokens": 200, "api_key": "sk-123"}
    assert _sanitize_cache_kwargs(params1) != _sanitize_cache_kwargs(params3)

@pytest.mark.parametrize("status, expected", [
    (429, True),   # Rate limit -> Retry
    (500, True),   # Server error -> Retry
    (503, True),   # Service unavail -> Retry
    (502, True),   # Bad Gateway -> Retry
    (504, True),   # Gateway Timeout -> Retry
    (400, False),  # Bad Request -> Fail
    (401, False),  # Auth -> Fail
    (404, False),  # Not Found -> Fail
])
def test_retry_predicates(status, expected):
    """Test the retry decision logic."""
    error = LLMAPIError("Test", status_code=status)
    assert _is_retriable_llm_api_error(error) is expected