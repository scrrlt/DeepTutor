# -*- coding: utf-8 -*-
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import logging
import pytest
from _pytest.logging import LogCaptureFixture

from src.services.embedding.adapters.azure import AzureEmbeddingAdapter
from src.services.embedding.adapters.base import EmbeddingRequest


@pytest.mark.asyncio
async def test_azure_embedding_adapter_embed() -> None:
    config = {
        "api_key": "test-key",
        "base_url": "https://test-resource.openai.azure.com/openai/deployments/test-deployment",
        "api_version": "2023-05-15",
        "model": "text-embedding-3-small",
        "dimensions": 1536,
    }
    adapter = AzureEmbeddingAdapter(config)

    request = EmbeddingRequest(
        texts=["hello world"], model="text-embedding-3-small"
    )

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "data": [{"embedding": [0.1] * 1536}],
        "model": "text-embedding-3-small",
        "usage": {"total_tokens": 2},
    }
    mock_response.raise_for_status = MagicMock()

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response

        response = await adapter.embed(request)

        assert response.embeddings == [[0.1] * 1536]
        assert response.model == "text-embedding-3-small"
        assert response.dimensions == 1536

        # Verify URL construction
        args, kwargs = mock_post.call_args
        url = args[0]
        assert (
            "https://test-resource.openai.azure.com/openai/deployments/test-deployment/embeddings"
            in url
        )
        assert "api-version=2023-05-15" in url

        # Verify headers
        headers = kwargs["headers"]
        assert headers["api-key"] == "test-key"


@pytest.mark.asyncio
async def test_azure_embedding_adapter_embed_http_error(
    caplog: LogCaptureFixture,
) -> None:
    caplog.set_level(logging.ERROR)
    config = {
        "api_key": "test-key",
        "base_url": "https://test-resource.openai.azure.com/openai/deployments/test-deployment",
        "model": "text-embedding-3-small",
    }
    adapter = AzureEmbeddingAdapter(config)
    request = EmbeddingRequest(
        texts=["error test"], model="text-embedding-3-small"
    )

    mock_response = MagicMock()
    mock_response.status_code = 401
    mock_response.text = "Unauthorized"
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "Unauthorized", request=MagicMock(), response=mock_response
    )

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response

        with pytest.raises(httpx.HTTPStatusError):
            await adapter.embed(request)

    assert "Azure API error (HTTP 401): Unauthorized" in caplog.text


@pytest.mark.asyncio
async def test_azure_embedding_adapter_embed_server_error(
    caplog: LogCaptureFixture,
) -> None:
    caplog.set_level(logging.ERROR)
    config = {
        "api_key": "test-key",
        "base_url": "https://test-resource.openai.azure.com/openai/deployments/test-deployment",
        "model": "text-embedding-3-small",
    }
    adapter = AzureEmbeddingAdapter(config)
    request = EmbeddingRequest(
        texts=["500 test"], model="text-embedding-3-small"
    )

    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.text = "Internal Server Error"
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "Internal Server Error", request=MagicMock(), response=mock_response
    )

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response

        with pytest.raises(httpx.HTTPStatusError):
            await adapter.embed(request)

    assert "Azure API error (HTTP 500): Internal Server Error" in caplog.text


@pytest.mark.asyncio
async def test_azure_embedding_adapter_embed_unexpected_error(
    caplog: LogCaptureFixture,
) -> None:
    caplog.set_level(logging.ERROR)
    config = {
        "api_key": "test-key",
        "base_url": "https://test.com",
        "model": "test-embedding-3-small",
    }
    adapter = AzureEmbeddingAdapter(config)
    request = EmbeddingRequest(texts=["fail"], model="any")

    with patch(
        "httpx.AsyncClient.post", side_effect=ValueError("Unexpected failure")
    ):
        with pytest.raises(ValueError, match="Unexpected failure"):
            await adapter.embed(request)

    assert (
        "Unexpected error during Azure embedding: Unexpected failure"
        in caplog.text
    )


@pytest.mark.asyncio
async def test_azure_embedding_adapter_empty_texts() -> None:
    config = {
        "api_key": "test-key",
        "base_url": "https://test-resource.openai.azure.com/openai/deployments/test-deployment",
    }
    adapter = AzureEmbeddingAdapter(config)
    request = EmbeddingRequest(texts=[], model="text-embedding-3-small")

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "data": [],
        "model": "text-embedding-3-small",
        "usage": {},
    }
    mock_response.raise_for_status = MagicMock()

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response
        with pytest.raises(
            ValueError,
            match="Invalid API response: missing or empty 'data' field",
        ):
            await adapter.embed(request)


@pytest.mark.asyncio
async def test_azure_embedding_adapter_batch_embed() -> None:
    config = {
        "api_key": "test-key",
        "base_url": "https://test-resource.openai.azure.com/openai/deployments/test-deployment",
    }
    adapter = AzureEmbeddingAdapter(config)
    request = EmbeddingRequest(
        texts=["text1", "text2"], model="text-embedding-3-small"
    )

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "data": [{"embedding": [0.1, 0.2]}, {"embedding": [0.3, 0.4]}],
        "model": "text-embedding-3-small",
        "usage": {},
    }
    mock_response.raise_for_status = MagicMock()

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response
        response = await adapter.embed(request)
        assert len(response.embeddings) == 2
        assert response.embeddings[0] == [0.1, 0.2]
        assert response.embeddings[1] == [0.3, 0.4]


@pytest.mark.asyncio
async def test_azure_embedding_adapter_missing_config() -> None:
    # Test missing api_key
    adapter = AzureEmbeddingAdapter({"base_url": "https://test.com"})
    request = EmbeddingRequest(texts=["hi"], model="m")
    with pytest.raises(ValueError, match="API key is required"):
        await adapter.embed(request)

    # Test missing base_url
    adapter = AzureEmbeddingAdapter({"api_key": "key"})
    with pytest.raises(ValueError, match="Base URL is required"):
        await adapter.embed(request)


@pytest.mark.asyncio
async def test_azure_embedding_adapter_timeout_handling() -> None:
    config = {
        "api_key": "key",
        "base_url": "https://test.com",
        "request_timeout": 42,
    }
    adapter = AzureEmbeddingAdapter(config)
    request = EmbeddingRequest(texts=["hi"], model="m")

    with patch("httpx.AsyncClient", autospec=True) as mock_client_class:
        mock_client = mock_client_class.return_value.__aenter__.return_value
        mock_client.post.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "data": [{"embedding": [0.1]}],
                "model": "m",
                "usage": {},
            },
        )

        await adapter.embed(request)

        # Check if AsyncClient was initialized with the correct timeout
        mock_client_class.assert_called_with(timeout=42)


@pytest.mark.asyncio
async def test_azure_embedding_adapter_malformed_json(
    caplog: LogCaptureFixture,
) -> None:
    caplog.set_level(logging.ERROR)
    config = {
        "api_key": "key",
        "base_url": "https://test.com",
    }
    adapter = AzureEmbeddingAdapter(config)
    request = EmbeddingRequest(texts=["hi"], model="m")

    mock_response = MagicMock()
    mock_response.status_code = 200
    # Response returning dict missing "data" key
    mock_response.json.return_value = {"something": "else"}
    mock_response.raise_for_status = MagicMock()

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response
        with pytest.raises(
            ValueError,
            match="Invalid API response: missing or empty 'data' field",
        ):
            await adapter.embed(request)


def test_azure_embedding_adapter_get_model_info() -> None:
    config = {
        "api_key": "test-key",
        "base_url": "https://test.com",
        "model": "text-embedding-3-small",
        "dimensions": 1536,
    }
    adapter = AzureEmbeddingAdapter(config)
    info = adapter.get_model_info()

    assert info["model"] == "text-embedding-3-small"
    assert info["dimensions"] == 1536
    assert info["provider"] == "azure"
    assert info["supports_variable_dimensions"] is True
