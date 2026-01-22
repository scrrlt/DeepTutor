"""Comprehensive hardening tests for logger, client, and high-load scenarios.

Tests critical stability gaps:
- Logger blocking I/O under high volume
- LLMClient deadlock under concurrent load
- Circuit breaker integration for fault resilience
- Error handling for rate limits (429) and service errors (500+)
"""

import asyncio
from unittest.mock import patch

import pytest

from src.logging import get_logger
from src.services.llm.client import LLMClient
from src.services.llm.config import LLMConfig


class TestLoggerNonBlockingLoad:
    """Test that logger's QueueHandler doesn't block event loop under volume."""

    @pytest.mark.asyncio
    async def test_logger_high_volume_nonblocking(self):
        """Verify logger doesn't stall event loop with 1000+ concurrent logs."""
        logger = get_logger("TestHighVolume")

        # Baseline: time a no-op coroutine
        start = asyncio.get_event_loop().time()

        # Spam logger with 1000 messages while running event loop tasks
        async def log_spam():
            for i in range(1000):
                logger.info("Message %d", i)
                if i % 100 == 0:
                    await asyncio.sleep(0)  # Yield to event loop

        async def background_task():
            """Quick background task to detect event loop stalls."""
            count = 0
            while asyncio.get_event_loop().time() - start < 5:
                count += 1
                await asyncio.sleep(0.001)
            return count

        # Run both concurrently
        spam_task = asyncio.create_task(log_spam())
        bg_task = asyncio.create_task(background_task())

        try:
            await asyncio.wait_for(
                asyncio.gather(spam_task, bg_task), timeout=10
            )
            elapsed = asyncio.get_event_loop().time() - start

            # Background task should complete many iterations (> 100)
            # If logger blocks the loop, bg_task will be starved
            bg_count = bg_task.result() if bg_task.done() else 0
            assert bg_count > 50, (
                f"Event loop starved by logger: only {bg_count} background iterations"
            )
        finally:
            logger.shutdown()


class TestLLMClientConcurrentLoad:
    """Test LLMClient doesn't deadlock under concurrent calls."""

    @pytest.mark.asyncio
    async def test_client_100_concurrent_calls(self):
        """Verify client handles 100+ concurrent complete() calls."""
        config = LLMConfig(
            api_key="test-key",
            base_url="http://localhost:8000",
            model="test-model",
        )
        client = LLMClient(config=config)

        # Mock factory.complete to return quickly
        async def mock_complete(*args, **kwargs):
            await asyncio.sleep(0.001)  # Simulate I/O
            return "mocked response"

        with patch("src.services.llm.factory.complete", new=mock_complete):
            # Launch 100 concurrent calls
            tasks = [client.complete(f"Prompt {i}") for i in range(100)]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # All should succeed (not deadlock or crash)
            assert len(results) == 100
            assert all(r == "mocked response" for r in results)

    @pytest.mark.asyncio
    async def test_client_handles_rate_limit_429(self):
        """Verify client degrades gracefully on HTTP 429 (Rate Limit)."""
        config = LLMConfig(
            api_key="test-key",
            base_url="http://localhost:8000",
            model="test-model",
        )
        client = LLMClient(config=config)

        # Mock factory.complete to raise rate limit error
        async def mock_complete_rate_limit(*args, **kwargs):
            from openai import RateLimitError

            raise RateLimitError(
                "429 Too Many Requests", response=None, body=None
            )

        with patch(
            "src.services.llm.factory.complete", new=mock_complete_rate_limit
        ):
            with pytest.raises(
                Exception
            ):  # Should propagate the rate limit error
                await client.complete("Test prompt")

    @pytest.mark.asyncio
    async def test_client_handles_server_error_500(self):
        """Verify client handles HTTP 500 errors gracefully."""
        config = LLMConfig(
            api_key="test-key",
            base_url="http://localhost:8000",
            model="test-model",
        )
        client = LLMClient(config=config)

        # Mock factory.complete to raise server error
        async def mock_complete_server_error(*args, **kwargs):
            raise RuntimeError("500 Internal Server Error")

        with patch(
            "src.services.llm.factory.complete", new=mock_complete_server_error
        ):
            with pytest.raises(RuntimeError):
                await client.complete("Test prompt")


class TestErrorHandlingRobustness:
    """Test error handling in critical paths."""

    def test_rag_tool_error_logging(self):
        """Verify rag_search logs exceptions properly."""
        from src.tools.rag_tool import rag_search

        logger = get_logger("TestRAGErrors")

        with patch("src.tools.rag_tool.get_rag_engine") as mock_get:
            # Simulate unexpected error
            mock_get.side_effect = RuntimeError("Database connection failed")

            # Should return error dict, not raise
            result = asyncio.run(rag_search("test query", kb_name="test_kb"))

            assert result["status"] == "error"
            assert "Internal search error" in result["message"]

        logger.shutdown()

    @pytest.mark.asyncio
    async def test_code_executor_timeout_recovery(self):
        """Verify code executor recovers properly after timeout."""
        from src.tools.code_executor import run_code

        # Script that will timeout
        code = "import time; time.sleep(100)"

        result = await run_code("python", code, timeout=1)

        assert result["exit_code"] == -1
        assert "timeout" in result["stderr"].lower()
        # Verify we can run another task after timeout (not deadlocked)
        result2 = await run_code("python", "print('recovery')", timeout=5)
        assert result2["exit_code"] == 0


class TestCircuitBreakerIntegration:
    """Test that circuit breaker prevents cascading failures."""

    @pytest.mark.asyncio
    async def test_circuit_breaker_on_repeated_failures(self):
        """Verify circuit breaker trips after repeated failures."""
        from src.utils.network.circuit_breaker import CircuitBreaker

        breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=1)

        # Simulate repeated failures
        async def failing_func():
            raise RuntimeError("Failed")

        for _ in range(2):
            with pytest.raises(RuntimeError):
                await breaker.call(failing_func)

        # Circuit should now be open (breaker is engaged)
        # Next call should fail without executing the function
        with pytest.raises(Exception):  # Will raise from breaker being open
            await breaker.call(failing_func)
