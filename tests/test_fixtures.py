import os
from unittest.mock import MagicMock

import pytest


def test_mock_env_variables_are_set(mock_env):
    """
    Test that environment variables are correctly mocked by the mock_env fixture.
    """
    assert os.getenv("NODE_ENV") == "test"
    assert os.getenv("LLM_BINDING") == "openai"
    assert os.getenv("LLM_MODEL") == "gpt-4o"
    assert os.getenv("LLM_API_KEY") == "sk-test-key"
    assert os.getenv("LLM_HOST") == "https://api.openai.com/v1"
    assert os.getenv("EMBEDDING_BINDING") == "openai"
    assert os.getenv("EMBEDDING_MODEL") == "text-embedding-3-large"
    assert os.getenv("EMBEDDING_API_KEY") == "sk-test-key"
    assert os.getenv("RAG_TOOL_MODULE_LOG_LEVEL") == "DEBUG"
    assert os.getenv("BACKEND_PORT") == "8001"
    assert os.getenv("SEARCH_PROVIDER") == "perplexity"
    assert os.getenv("SEARCH_API_KEY") == "test-search-key"


def test_mock_openai_client_is_mocked(mock_openai_client):
    """
    Test that the openai.AsyncOpenAI client is correctly mocked.
    """
    assert isinstance(mock_openai_client, MagicMock)
    assert isinstance(mock_openai_client.chat.completions.create, MagicMock)
    # Ensure that calling the mocked method does not raise an error
    mock_openai_client.chat.completions.create("test", "test")
    mock_openai_client.chat.completions.create.assert_called_once_with("test", "test")


@pytest.mark.asyncio
async def test_mock_openai_client_async_usage(mock_openai_client):
    """
    Test that the mocked openai.AsyncOpenAI client can be used in an async context.
    """
    # Simulate an async call to the mocked client
    await mock_openai_client.chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": "Hello"}]
    )
    mock_openai_client.chat.completions.create.assert_called_once_with(
        model="gpt-4o", messages=[{"role": "user", "content": "Hello"}]
    )
