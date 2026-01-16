# tests/services/llm/test_cloud_provider.py
import pytest
import httpx
from unittest.mock import MagicMock, patch, AsyncMock
from src.services.llm.cloud_provider import OpenAIProvider
from src.services.llm.exceptions import LLMRateLimitError
from src.config.config import LLMConfig

@pytest.fixture
def llm_config():
    return LLMConfig(
        model="gpt-4o",
        api_key="sk-test",
        binding="openai"
    )

@pytest.fixture
def openai_provider(llm_config):
    # Patch the shared HTTP client getter
    with patch("src.services.llm.cloud_provider.get_shared_http_client", new_callable=AsyncMock) as mock_get_client:
        mock_http = AsyncMock(spec=httpx.AsyncClient)
        mock_get_client.return_value = mock_http
        
        provider = OpenAIProvider(config=llm_config)
        # Manually inject client to bypass lock logic in test
        provider.client = AsyncMock() 
        provider.client.chat.completions.create = AsyncMock()
        return provider

@pytest.mark.asyncio
async def test_complete_success(openai_provider):
    # Setup Mock Response
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Success"
    mock_response.usage.model_dump.return_value = {"total_tokens": 10}
    
    openai_provider.client.chat.completions.create.return_value = mock_response
    
    result = await openai_provider.complete("Hello")
    
    assert result.content == "Success"
    assert result.usage["total_tokens"] == 10

@pytest.mark.asyncio
async def test_error_mapping_rate_limit(openai_provider):
    # Simulate OpenAI 429 Error
    import openai
    error = openai.RateLimitError("Rate limit exceeded", response=MagicMock(status_code=429), body=None)
    
    openai_provider.client.chat.completions.create.side_effect = error
    
    # We expect our custom LLMRateLimitError to be raised
    with pytest.raises(LLMRateLimitError):
        # Call complete(), which should map the underlying OpenAI RateLimitError
        # to our custom LLMRateLimitError.
        await openai_provider.complete("Hello")
@pytest.mark.asyncio
async def test_token_param_resolution(openai_provider):
    """Ensure o1/o3 models use max_completion_tokens."""
    openai_provider.config.model = "o1-preview"
    
    await openai_provider.complete("Test")
    
    call_kwargs = openai_provider.client.chat.completions.create.call_args.kwargs
    assert "max_completion_tokens" in call_kwargs
    assert "max_tokens" not in call_kwargs