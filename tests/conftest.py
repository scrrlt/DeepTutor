import os
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(scope="session", autouse=True)
def mock_env() -> Generator[None, None, None]:
    """
    Mock environment variables for the entire test session.

    This ensures that tests do not accidentally rely on the local environment
    configuration or .env files, providing a deterministic test environment.
    """
    env_vars = {
        "NODE_ENV": "test",
        "LLM_BINDING": "openai",
        "LLM_MODEL": "gpt-4o",
        "LLM_API_KEY": "sk-test-key",
        "LLM_HOST": "https://api.openai.com/v1",
        "EMBEDDING_BINDING": "openai",
        "EMBEDDING_MODEL": "text-embedding-3-large",
        "EMBEDDING_API_KEY": "sk-test-key",
        "RAG_TOOL_MODULE_LOG_LEVEL": "DEBUG",
        "BACKEND_PORT": "8001",
        "SEARCH_PROVIDER": "perplexity",
        "SEARCH_API_KEY": "test-search-key",
    }
    with patch.dict(os.environ, env_vars):
        yield


@pytest.fixture
def mock_openai_client() -> Generator[MagicMock, None, None]:
    """
    Fixture to mock the OpenAI asynchronous client.

    Returns:
        MagicMock: A mock object replacing the AsyncOpenAI client.
    """
    with patch("openai.AsyncOpenAI") as mock_client:
        mock_instance = mock_client.return_value
        mock_instance.chat.completions.create = MagicMock()
        yield mock_instance