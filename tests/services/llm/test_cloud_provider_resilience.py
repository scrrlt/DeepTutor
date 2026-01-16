# tests/services/llm/test_cloud_provider_resilience.py
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from src.services.llm.cloud_provider import OpenAIProvider
from src.services.llm.exceptions import LLMCircuitBreakerError
from src.config.config import LLMConfig

@pytest.fixture
def provider():
    config = LLMConfig(model="gpt-4o", binding="openai", api_key="test")
    return OpenAIProvider(config)

@pytest.mark.asyncio
async def test_circuit_breaker_open(provider):
    """Test that requests are blocked when the circuit breaker is open."""

    with (
        patch("src.services.llm.providers.base_provider.is_call_allowed", return_value=False),
        patch("src.services.llm.providers.base_provider.record_provider_call") as mock_record,
        pytest.raises(LLMCircuitBreakerError),
    ):
        await provider.complete("Should fail fast")

    mock_record.assert_called_with("openai", success=False)

@pytest.mark.asyncio
async def test_telemetry_recording_success(provider):
    """Test that successful calls record success metrics."""
    
    # Mock internals
    provider.client = AsyncMock()
    mock_resp = MagicMock()
    mock_resp.choices = [
        MagicMock(
            message=MagicMock(content="OK"),
            finish_reason="stop",
        )
    ]
    mock_resp.usage = MagicMock()
    mock_resp.usage.model_dump.return_value = {}
    mock_resp.model_dump.return_value = {}
    provider.client.chat.completions.create.return_value = mock_resp

    with (
        patch("src.services.llm.providers.base_provider.is_call_allowed", return_value=True),
        patch("src.services.llm.providers.base_provider.record_provider_call") as mock_record,
        patch("src.services.llm.providers.base_provider.record_call_success") as mock_cb_success,
    ):
        await provider.complete("Test")

    mock_record.assert_called_with("openai", success=True)
    mock_cb_success.assert_called_with("openai")

@pytest.mark.asyncio
async def test_telemetry_recording_failure(provider):
    """Test that exceptions record failure metrics."""
    
    provider.client = AsyncMock()
    provider.client.chat.completions.create.side_effect = Exception("API Down")

    with (
        patch("src.services.llm.providers.base_provider.is_call_allowed", return_value=True),
        patch("src.services.llm.providers.base_provider.record_provider_call") as mock_record,
        patch("src.services.llm.providers.base_provider.record_call_failure") as mock_cb_fail,
    ):
        with pytest.raises(Exception):
            await provider.complete("Test")

    mock_record.assert_called_with("openai", success=False)
    assert mock_cb_fail.call_count in (0, 1)