# -*- coding: utf-8 -*-
import os
import pytest
import importlib

adapters = importlib.import_module("src.services.embedding.adapters.cohere")
EmbeddingRequest = adapters.EmbeddingRequest
CohereAdapter = adapters.CohereEmbeddingAdapter


@pytest.mark.integration
@pytest.mark.asyncio
async def test_cohere_embedding_integration():
    api_key = os.getenv("COHERE_API_KEY")
    if not api_key:
        pytest.skip("COHERE_API_KEY not set")

    adapter = CohereAdapter({"base_url": os.getenv("COHERE_API_URL"), "api_key": api_key})
    req = EmbeddingRequest(texts=["hello world"], model=adapter.model or "small")

    resp = await adapter.embed(req)
    assert resp.embeddings and isinstance(resp.embeddings[0], list)
    assert resp.dimensions == len(resp.embeddings[0])
