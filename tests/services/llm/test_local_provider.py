# tests/services/llm/test_local_provider.py
import pytest
import httpx
from unittest.mock import MagicMock, patch, AsyncMock
from src.services.llm.local_provider import LocalLLMProvider
from src.config.config import LLMConfig

@pytest.fixture
def local_config():
    return LLMConfig(
        model="llama3",
        base_url="http://localhost:11434/v1",
        binding="ollama",
        api_key="ollama"  # Often ignored but good to have
    )

@pytest.fixture
def local_provider(local_config):
    # Mock the shared HTTP client getter
    with patch("src.services.llm.local_provider.get_shared_http_client", new_callable=AsyncMock) as mock_get_client:
        mock_http = AsyncMock(spec=httpx.AsyncClient)
        mock_get_client.return_value = mock_http
        
        provider = LocalLLMProvider(config=local_config)
        # Inject mocks manually
        provider.client = AsyncMock() 
        provider.client.chat.completions.create = AsyncMock()
        return provider

@pytest.mark.asyncio
async def test_ollama_completion(local_provider):
    # Setup Mock Response (Ollama follows OpenAI format usually)
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "I am Llama."
    mock_response.usage.model_dump.return_value = {"total_tokens": 15}
    
    local_provider.client.chat.completions.create.return_value = mock_response
    
    result = await local_provider.complete("Who are you?")
    
    # Assertions
    assert result.content == "I am Llama."
    assert result.provider == "ollama"
    
    # Verify the call passed the correct URL and model
    call_kwargs = local_provider.client.chat.completions.create.call_args.kwargs
    assert call_kwargs["model"] == "llama3"

@pytest.mark.asyncio
async def test_local_connection_error(local_provider):
    """Ensure we catch connection refused errors from local servers."""
    import openai
    
    # Simulate server down
    error = openai.APIConnectionError(message="Connection refused", request=MagicMock())
    local_provider.client.chat.completions.create.side_effect = error
    
    # Should propagate as a mapped exception or standard error
    # checking if mapping works
    with pytest.raises(Exception) as excinfo:
        await local_provider.complete("Hello")
    
    # Depending on your error_mapping.py, this might be LLMConnectionError
    # For now, just ensure it doesn't hang silently
    assert "Connection refused" in str(excinfo.value)

@pytest.mark.asyncio
async def test_streaming_chunks(local_provider):
    """Test handling of stream chunks from local provider."""
    # Mock the stream generator
    async def mock_stream(*args, **kwargs):
        chunk1 = MagicMock()
        chunk1.choices[0].delta.content = "Hello "
        yield chunk1
        
        chunk2 = MagicMock()
        chunk2.choices[0].delta.content = "World"
        yield chunk2

    local_provider.client.chat.completions.create.side_effect = mock_stream
    
    chunks = []
    async for chunk in local_provider.stream("Hi"):
        chunks.append(chunk.content)
        
    assert "".join(chunks) == "Hello World"