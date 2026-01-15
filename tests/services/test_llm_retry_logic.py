"""
Unit Tests for LLM Factory Retry Logic
======================================

Tests the centralized retry mechanism in factory.py to ensure:
1. Retry logic is NOT nested (no 3x3=9 attempts)
2. Retriable errors are correctly identified (5xx, 429)
3. Non-retriable errors fail fast (4xx except 429)
4. Circuit breaker integration works correctly
"""

from unittest.mock import AsyncMock, patch

import pytest

from src.services.llm import cloud_provider
from src.services.llm.exceptions import (
    LLMAPIError,
    LLMAuthenticationError,
    LLMError,
    LLMRateLimitError,
    LLMTimeoutError,
)
from src.services.llm.factory import _is_retriable_error, complete


class TestRetriableErrorDetection:
    """Test _is_retriable_error function for correct error classification."""

    def test_timeout_error_is_retriable(self):
        """Timeout errors should be retried."""
        error = LLMTimeoutError("Request timed out")
        assert _is_retriable_error(error) is True

    def test_rate_limit_error_is_retriable(self):
        """Rate limit errors (429) should be retried."""
        error = LLMRateLimitError("Rate limit exceeded")
        assert _is_retriable_error(error) is True

    def test_authentication_error_not_retriable(self):
        """Authentication errors (401) should NOT be retried."""
        error = LLMAuthenticationError("Invalid API key")
        assert _is_retriable_error(error) is False

    def test_server_error_5xx_is_retriable(self):
        """Server errors (5xx) should be retried."""
        for status_code in [500, 502, 503, 504]:
            error = LLMAPIError("Server error", status_code=status_code)
            assert _is_retriable_error(error) is True, (
                f"Status {status_code} should be retriable"
            )

    def test_client_error_4xx_not_retriable(self):
        """Client errors (4xx except 429) should NOT be retried."""
        for status_code in [400, 401, 403, 404]:
            error = LLMAPIError("Client error", status_code=status_code)
            assert _is_retriable_error(error) is False, (
                f"Status {status_code} should not be retriable"
            )

    def test_429_rate_limit_via_api_error_is_retriable(self):
        """429 rate limit via LLMAPIError should be retried."""
        error = LLMAPIError("Rate limit", status_code=429)
        assert _is_retriable_error(error) is True


class TestProviderInterfaces:
    """Test that provider interfaces expose expected call methods."""

    def test_cloud_provider_has_complete_and_stream(self):
        """Cloud provider must implement complete and stream."""
        assert hasattr(cloud_provider, "complete")
        assert hasattr(cloud_provider, "stream")


class TestNoNestedRetries:
    """Test that retry logic is not nested between factory and provider."""
    
    @pytest.mark.asyncio
    async def test_provider_execute_called_once_per_factory_retry(self):
        """
        Verify that provider.execute is called exactly once per factory retry,
        not multiple times (which would indicate nested retries).
        """
        with patch("src.services.llm.factory.cloud_provider") as mock_cloud:
            # Mock provider to fail twice, then succeed
            mock_cloud.complete = AsyncMock(side_effect=[
                LLMRateLimitError("Rate limit"),
                LLMRateLimitError("Rate limit"),
                "Success"
            ])
            
            result = await complete(
                prompt="test",
                model="gpt-4",
                base_url="https://api.openai.com/v1",
                api_key="test-key",
                max_retries=3,
                retry_delay=0.01,  # Fast retry for testing
            )
            
            assert result == "Success"
            # Should be called exactly 3 times (2 failures + 1 success)
            # NOT 9 times (3 factory retries × 3 provider retries)
            assert mock_cloud.complete.call_count == 3
    
    @pytest.mark.asyncio
    async def test_non_retriable_error_fails_immediately(self):
        """Non-retriable errors should fail without retry."""
        with patch("src.services.llm.factory.cloud_provider") as mock_cloud:
            mock_cloud.complete = AsyncMock(side_effect=LLMAuthenticationError("Invalid key"))
            
            with pytest.raises(LLMAuthenticationError):
                await complete(
                    prompt="test",
                    model="gpt-4",
                    base_url="https://api.openai.com/v1",
                    api_key="bad-key",
                    max_retries=3,
                )
            
            # Should only be called once (no retries)
            assert mock_cloud.complete.call_count == 1


class TestCircuitBreakerIntegration:
    """Test circuit breaker behavior in provider execution."""
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_open_raises_llm_error(self):
        """When circuit breaker is open, should raise LLMError immediately."""
        with patch("src.services.llm.providers.base_provider.is_call_allowed") as mock_allowed:
            mock_allowed.return_value = False
            
            from src.services.llm.providers.base_provider import BaseLLMProvider
            from src.services.llm.config import LLMConfig
            
            # Create a test provider
            config = LLMConfig(
                provider_name="test",
                api_key="test-key",
                model="test-model",
                base_url="https://test.com",
            )
            provider = BaseLLMProvider(config)
            
            async def test_func():
                return "should not be called"
            
            with pytest.raises(LLMError, match="Circuit breaker open"):
                await provider.execute(test_func)
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_closed_executes_normally(self):
        """When circuit breaker is closed, should execute function normally."""
        with patch("src.services.llm.providers.base_provider.is_call_allowed") as mock_allowed, \
             patch("src.services.llm.providers.base_provider.record_provider_call") as mock_record, \
             patch("src.services.llm.providers.base_provider.record_call_success") as mock_success:
            
            mock_allowed.return_value = True
            
            from src.services.llm.providers.base_provider import BaseLLMProvider
            from src.services.llm.config import LLMConfig
            
            config = LLMConfig(
                provider_name="test",
                api_key="test-key",
                model="test-model",
                base_url="https://test.com",
            )
            provider = BaseLLMProvider(config)
            
            async def test_func():
                return "success"
            
            result = await provider.execute(test_func)
            
            assert result == "success"
            mock_record.assert_called_with("test", success=True)
            mock_success.assert_called_once_with("test")


class TestExponentialBackoff:
    """Test exponential backoff behavior in retries."""
    
    @pytest.mark.asyncio
    async def test_retry_delays_increase_exponentially(self):
        """Verify that retry delays increase exponentially."""
        with patch("src.services.llm.factory.cloud_provider") as mock_cloud:
            call_count = 0
            sleep_calls = []
            mock_sleep = AsyncMock(side_effect=sleep_calls.append)

            async def track_time(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count < 3:
                    raise LLMTimeoutError("Timeout")
                return "Success"

            mock_cloud.complete = AsyncMock(side_effect=track_time)

            await complete(
                prompt="test",
                model="gpt-4",
                base_url="https://api.openai.com/v1",
                api_key="test-key",
                max_retries=3,
                retry_delay=0.1,
                sleep=mock_sleep,
            )

            assert call_count == 3
            assert mock_sleep.await_count == 2
            assert len(sleep_calls) == 2
            delay1 = sleep_calls[0]
            delay2 = sleep_calls[1]
            assert delay2 > delay1, "Exponential backoff not working"


class TestOpenAIStreamingErrors:
    """Test handling of OpenAI streaming errors."""

    @pytest.mark.asyncio
    async def test_stream_connection_error_is_raised(self):
        """Ensure stream connection errors are logged and re-raised."""
        import openai

        from src.services.llm.config import LLMConfig
        from src.services.llm.providers.open_ai import OpenAIProvider

        async def error_stream():
            raise openai.APIConnectionError("Stream failed")
            yield  # pragma: no cover

        config = LLMConfig(
            provider_name="openai",
            api_key="test-key",
            model="gpt-4",
            base_url="https://api.openai.com/v1",
        )
        provider = OpenAIProvider(config)

        with (
            patch.object(provider, "execute", return_value=error_stream()),
            patch("src.services.llm.providers.open_ai.logger") as mock_logger,
        ):
            with pytest.raises(openai.APIConnectionError):
                async for _ in provider.stream("hello"):
                    pass

            mock_logger.error.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
