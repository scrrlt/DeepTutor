#!/usr/bin/env python

"""
Tests for the LLM factory and routing helpers.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.services.llm.config import LLMConfig
from src.services.llm.factory import LLMFactory, complete, fetch_models, stream
from src.services.llm.providers.base_provider import BaseLLMProvider


@patch("src.services.llm.providers.routing.register_provider")
def test_llm_factory_get_provider(mock_register_provider):
    """
    Tests that the LLMFactory.get_provider method correctly creates a provider instance.
    """
    # Create a mock for the RoutingProvider class itself
    with patch("src.services.llm.providers.routing.RoutingProvider") as mock_routing_provider_cls:
        mock_provider_instance = MagicMock(spec=BaseLLMProvider)
        mock_routing_provider_cls.return_value = mock_provider_instance

        # LLMConfig requires api_key
        config = LLMConfig(model="test_model", binding="test_binding", api_key="sk-test")

        provider = LLMFactory.get_provider(config)

        mock_routing_provider_cls.assert_called_with(config)
        assert provider is mock_provider_instance


@pytest.mark.asyncio
@patch("src.services.llm.factory.get_llm_config")
@patch("src.services.llm.factory.cloud_provider.complete", new_callable=AsyncMock)
async def test_complete(mock_cloud_complete, mock_get_config):
    """
    Tests that the complete function correctly calls the provider's complete method.
    """
    mock_get_config.return_value = LLMConfig(
        model="test_model", base_url="https://api.openai.com/v1", api_key="sk-test"
    )
    mock_cloud_complete.return_value = "test_response"

    response = await complete("test_prompt")

    mock_cloud_complete.assert_called_once()
    assert response == "test_response"


@pytest.mark.asyncio
@patch("src.services.llm.factory.get_llm_config")
@patch("src.services.llm.factory.cloud_provider.stream")
async def test_stream(mock_cloud_stream, mock_get_config):
    """
    Tests that the stream function correctly calls the provider's stream method.
    """
    mock_get_config.return_value = LLMConfig(
        model="test_model", base_url="https://api.openai.com/v1", api_key="sk-test"
    )

    async def mock_stream_generator():
        yield "test"
        yield " response"

    mock_cloud_stream.return_value = mock_stream_generator()

    chunks = []
    async for chunk in stream("test_prompt"):
        chunks.append(chunk)

    mock_cloud_stream.assert_called_once()
    assert "".join(chunks) == "test response"


@pytest.mark.asyncio
@patch(
    "src.services.llm.local_provider.fetch_models",
    new_callable=AsyncMock,
)
@patch(
    "src.services.llm.cloud_provider.fetch_models",
    new_callable=AsyncMock,
)
async def test_fetch_models(mock_cloud_fetch, mock_local_fetch):
    """
    Tests that the fetch_models function correctly calls the correct provider's fetch_models method.
    """
    # Test local provider
    await fetch_models("ollama", "http://localhost:11434/v1", "sk-dummy")
    mock_local_fetch.assert_called_once_with("http://localhost:11434/v1", "sk-dummy")

    # Test cloud provider
    await fetch_models("openai", "https://api.openai.com/v1", "sk-dummy")
    mock_cloud_fetch.assert_called_once_with("https://api.openai.com/v1", "sk-dummy", "openai")
