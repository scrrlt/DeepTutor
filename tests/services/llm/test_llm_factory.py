#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests for the LLM factory and routing helpers.
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from src.services.llm.factory import LLMFactory, complete, stream, fetch_models
from src.services.llm.config import LLMConfig
from src.services.llm.providers.base_provider import BaseLLMProvider
from src.services.llm.types import LLMResponse, StreamChunk


@patch('src.services.llm.registry.get_provider_class')
def test_llm_factory_get_provider(mock_get_provider_class):
    """
    Tests that the LLMFactory.get_provider method correctly creates a provider instance.
    """
    mock_provider_cls = MagicMock()
    mock_get_provider_class.return_value = mock_provider_cls
    config = LLMConfig(model="test_model", binding="test_binding")

    provider = LLMFactory.get_provider(config)

    mock_get_provider_class.assert_called_with("routing")
    mock_provider_cls.assert_called_with(config)
    assert isinstance(provider, BaseLLMProvider)


@pytest.mark.asyncio
@patch('src.services.llm.factory.LLMFactory.get_provider')
async def test_complete(mock_get_provider):
    """
    Tests that the complete function correctly calls the provider's complete method.
    """
    mock_provider = MagicMock(spec=BaseLLMProvider)
    mock_provider.complete = AsyncMock(return_value=LLMResponse(content="test_response"))
    mock_get_provider.return_value = mock_provider

    response = await complete("test_prompt")

    mock_provider.complete.assert_called_once()
    assert response == "test_response"


@pytest.mark.asyncio
@patch('src.services.llm.factory.LLMFactory.get_provider')
async def test_stream(mock_get_provider):
    """
    Tests that the stream function correctly calls the provider's stream method.
    """
    async def mock_stream_method(*args, **kwargs):
        yield StreamChunk(delta="test")
        yield StreamChunk(delta=" response")
    
    mock_provider = MagicMock(spec=BaseLLMProvider)
    mock_provider.stream = mock_stream_method
    mock_get_provider.return_value = mock_provider

    chunks = []
    async for chunk in stream("test_prompt"):
        chunks.append(chunk)

    assert "".join(chunks) == "test response"


@pytest.mark.asyncio
@patch('src.services.llm.factory.local_provider.fetch_models', new_callable=AsyncMock)
@patch('src.services.llm.factory.cloud_provider.fetch_models', new_callable=AsyncMock)
async def test_fetch_models(mock_cloud_fetch, mock_local_fetch):
    """
    Tests that the fetch_models function correctly calls the correct provider's fetch_models method.
    """
    # Test local provider
    await fetch_models("ollama", "http://localhost:11434/v1")
    mock_local_fetch.assert_called_once_with("http://localhost:11434/v1", None)

    # Test cloud provider
    await fetch_models("openai", "https://api.openai.com/v1")
    mock_cloud_fetch.assert_called_once_with("https://api.openai.com/v1", None, "openai")
