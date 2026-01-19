# -*- coding: utf-8 -*-
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import logging
import pytest
from _pytest.logging import LogCaptureFixture

from src.services.embedding.adapters.cohere import CohereEmbeddingAdapter
from src.services.embedding.adapters.base import EmbeddingRequest


@pytest.mark.asyncio
async def test_cohere_embedding_adapter_embed() -> None:
    config = {
        "api_key": "test-key",
        "base_url": "https://api.cohere.ai",
        "model": "embed-v4.0",
    }
    adapter = CohereEmbeddingAdapter(config)
    request = EmbeddingRequest(texts=["hello"], model="embed-v4.0")

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "embeddings": {"float": [[0.1, 0.2]]},
        "model": "embed-v4.0",
        "meta": {"billed_units": {"input_tokens": 1}},
    }
    mock_response.raise_for_status = MagicMock()

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response
        response = await adapter.embed(request)
        assert response.embeddings == [[0.1, 0.2]]
        assert response.dimensions == 2


@pytest.mark.asyncio
async def test_cohere_embedding_adapter_http_error(
    caplog: LogCaptureFixture,
) -> None:
    caplog.set_level(logging.ERROR)
    config = {
        "api_key": "key",
        "base_url": "https://api.cohere.ai",
        "model": "m",
    }
    adapter = CohereEmbeddingAdapter(config)
    request = EmbeddingRequest(texts=["err"], model="m")

    mock_response = MagicMock()
    mock_response.status_code = 400
    mock_response.text = "Bad Request"
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "Err", request=MagicMock(), response=mock_response
    )

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response
        with pytest.raises(httpx.HTTPStatusError):
            await adapter.embed(request)
    assert "HTTP 400 response body: Bad Request" in caplog.text
