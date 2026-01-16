"""
Integration Tests for Circuit Breaker Exception Handler
=======================================================

Tests the global exception handler in API main.py to ensure:
1. Circuit breaker errors return 503 Service Unavailable
2. Rate limit errors return 429 Too Many Requests
3. Authentication errors return 401 Unauthorized
4. Timeout errors return 504 Gateway Timeout
5. Generic API errors return appropriate status codes
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

from src.services.llm.exceptions import (
    LLMAPIError,
    LLMAuthenticationError,
    LLMError,
    LLMRateLimitError,
    LLMTimeoutError,
)


@pytest.fixture
def test_app():
    """Create a test FastAPI app with the exception handler."""
    from fastapi import FastAPI
    from src.api.main import llm_error_handler
    
    app = FastAPI()
    app.add_exception_handler(LLMError, llm_error_handler)
    
    @app.get("/test/circuit-breaker")
    async def test_circuit_breaker():
        error = LLMError("Circuit breaker open for provider openai")
        setattr(error, "is_circuit_breaker", True)
        raise error
    
    @app.get("/test/rate-limit")
    async def test_rate_limit():
        raise LLMRateLimitError("Rate limit exceeded")
    
    @app.get("/test/auth")
    async def test_auth():
        raise LLMAuthenticationError("Invalid API key")
    
    @app.get("/test/timeout")
    async def test_timeout():
        raise LLMTimeoutError("Request timed out")
    
    @app.get("/test/api-error/{status_code}")
    async def test_api_error(status_code: int):
        raise LLMAPIError(f"API error {status_code}", status_code=status_code)
    
    @app.get("/test/generic-error")
    async def test_generic_error():
        raise ValueError("Generic error")
    
    return app


class TestLLMExceptionHandler:
    """Test LLM exception handler returns correct status codes."""
    
    def test_circuit_breaker_returns_503(self, test_app):
        """Circuit breaker errors should return 503 Service Unavailable."""
        client = TestClient(test_app, raise_server_exceptions=False)
        response = client.get("/test/circuit-breaker")
        
        assert response.status_code == 503
        data = response.json()
        assert data["error"] == "service_unavailable"
        assert "circuit breaker" in data["message"].lower()
    
    def test_rate_limit_returns_429(self, test_app):
        """Rate limit errors should return 429 Too Many Requests."""
        client = TestClient(test_app, raise_server_exceptions=False)
        response = client.get("/test/rate-limit")
        
        assert response.status_code == 429
        data = response.json()
        assert data["error"] == "rate_limit_exceeded"
    
    def test_authentication_returns_401(self, test_app):
        """Authentication errors should return 401 Unauthorized."""
        client = TestClient(test_app, raise_server_exceptions=False)
        response = client.get("/test/auth")
        
        assert response.status_code == 401
        data = response.json()
        assert data["error"] == "authentication_failed"
    
    def test_timeout_returns_504(self, test_app):
        """Timeout errors should return 504 Gateway Timeout."""
        client = TestClient(test_app, raise_server_exceptions=False)
        response = client.get("/test/timeout")
        
        assert response.status_code == 504
        data = response.json()
        assert data["error"] == "gateway_timeout"
    
    def test_api_error_maps_status_code(self, test_app):
        """LLMAPIError should return the embedded status code."""
        client = TestClient(test_app)
        
        # Test various status codes
        for status_code in [400, 500, 502, 503]:
            response = client.get(f"/test/api-error/{status_code}")
            assert response.status_code == status_code
            data = response.json()
            assert data["error"] == "llm_api_error"
    
    def test_generic_error_re_raised(self, test_app):
        """Non-LLM errors should be re-raised for default handling."""
        client = TestClient(test_app, raise_server_exceptions=False)
        
        # Generic errors should raise 500 (FastAPI default)
        response = client.get("/test/generic-error")
        assert response.status_code == 500


class TestCircuitBreakerLogging:
    """Test that circuit breaker errors are properly logged."""
    
    def test_circuit_breaker_logged_as_warning(self, test_app):
        """Circuit breaker triggers should be logged as warnings."""
        with patch("src.api.main.logger") as mock_logger:
            client = TestClient(test_app, raise_server_exceptions=False)
            client.get("/test/circuit-breaker")
            
            # Verify warning was logged
            mock_logger.warning.assert_called_once()
            call_args = mock_logger.warning.call_args[0][0]
            assert "circuit breaker" in call_args.lower()
    
    def test_rate_limit_logged_as_warning(self, test_app):
        """Rate limit errors should be logged as warnings."""
        with patch("src.api.main.logger") as mock_logger:
            client = TestClient(test_app, raise_server_exceptions=False)
            client.get("/test/rate-limit")
            
            mock_logger.warning.assert_called_once()
            call_args = mock_logger.warning.call_args[0][0]
            assert "rate limit" in call_args.lower()
    
    def test_auth_error_logged_as_error(self, test_app):
        """Authentication errors should be logged as errors."""
        with patch("src.api.main.logger") as mock_logger:
            client = TestClient(test_app, raise_server_exceptions=False)
            client.get("/test/auth")
            
            mock_logger.error.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
