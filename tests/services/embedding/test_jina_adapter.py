# -*- coding: utf-8 -*-
import logging
from unittest.mock import AsyncMock, MagicMock, patch

from _pytest.logging import LogCaptureFixture
import httpx
import pytest

from src.services.embedding.adapters.base import EmbeddingRequest
from src.services.embedding.adapters.jina import JinaEmbeddingAdapter


@pytest.mark.asyncio
async def test_jina_embedding_adapter_embed() -> None:
    config = {
        "api_key": "test-key",
        "base_url": "https://api.jina.ai/v1",
        "model": "jina-embeddings-v3",
    }
    adapter = JinaEmbeddingAdapter(config)
    request = EmbeddingRequest(texts=["hello"], model="jina-embeddings-v3")

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "data": [{"embedding": [0.1, 0.2]}],
        "model": "jina-embeddings-v3",
        "usage": {"total_tokens": 1},
    }
    mock_response.raise_for_status = MagicMock()

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response
        response = await adapter.embed(request)
        assert response.embeddings == [[0.1, 0.2]]
        assert response.dimensions == 2


@pytest.mark.asyncio
async def test_jina_embedding_adapter_http_error(
    caplog: LogCaptureFixture,
) -> None:
    caplog.set_level(logging.ERROR)
    config = {
        "api_key": "key",
        "base_url": "https://api.jina.ai/v1",
        "model": "m",
    }
    adapter = JinaEmbeddingAdapter(config)
    request = EmbeddingRequest(texts=["err"], model="m")

    mock_response = MagicMock()
    mock_response.status_code = 403
    mock_response.text = "Forbidden"
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "Err", request=MagicMock(), response=mock_response
    )

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response
        with pytest.raises(httpx.HTTPStatusError):
            await adapter.embed(request)
    assert "HTTP 403 response body: Forbidden" in caplog.text
