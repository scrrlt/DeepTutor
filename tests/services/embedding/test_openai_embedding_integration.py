# -*- coding: utf-8 -*-
import os
import pytest
import importlib

adapters = importlib.import_module("src.services.embedding.adapters.openai_compatible")
EmbeddingRequest = adapters.EmbeddingRequest
OpenAIAdapter = adapters.OpenAICompatibleEmbeddingAdapter


@pytest.mark.integration
@pytest.mark.asyncio
async def test_openai_embedding_integration():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set")

    adapter = OpenAIAdapter({"base_url": "https://api.openai.com", "api_key": api_key})

    req = EmbeddingRequest(texts=["hello world"], model=adapter.model or "text-embedding-3-small")
    try:
        resp = await adapter.embed(req)
    except Exception as e:
        # Network or SSL problems may occur in some CI environments; treat as skipped
        import ssl

        if isinstance(e, ssl.SSLError):
            pytest.skip(f"SSL error during integration test: {e}")
        raise

    assert resp.embeddings and isinstance(resp.embeddings[0], list)
    assert resp.dimensions == len(resp.embeddings[0])
