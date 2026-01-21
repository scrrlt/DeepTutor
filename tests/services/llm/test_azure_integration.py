# -*- coding: utf-8 -*-
"""Integration tests for Azure OpenAI service (chat and embeddings).

These tests are marked as integration and will be skipped unless the required
environment variables are present. They make real network calls and should be
controlled via CI secrets or local .env files during manual runs.
"""

import os
import pytest

from src.services.llm import factory
from src.services.embedding.client import EmbeddingClient


@pytest.mark.integration
@pytest.mark.asyncio
async def test_azure_openai_chat_real():
    api_key = os.getenv("AZURE_API_KEY")
    base_url = os.getenv("AZURE_BASE_URL") or os.getenv("LLM_HOST")
    api_version = os.getenv("AZURE_API_VERSION") or os.getenv("LLM_API_VERSION")
    if not (api_key and base_url and api_version):
        pytest.skip(
            "Azure OpenAI integration config not set (AZURE_API_KEY/AZURE_BASE_URL/AZURE_API_VERSION)"
        )

    response = await factory.complete(
        prompt="Say 'Azure Ready'",
        model=os.getenv("AZURE_MODEL", "gpt-35-turbo"),
        api_key=api_key,
        base_url=base_url,
        api_version=api_version,
        binding="azure_openai",
    )

    assert "ready" in response.lower()


@pytest.mark.integration
def test_azure_embedding_real():
    # EmbeddingClient reads EMBEDDING_API_KEY/EMBEDDING_HOST or falls back to Azure env vars
    api_key = os.getenv("AZURE_API_KEY") or os.getenv("EMBEDDING_API_KEY")
    base_url = os.getenv("EMBEDDING_HOST") or os.getenv("AZURE_BASE_URL")
    if not (api_key and base_url):
        pytest.skip("Azure embedding config not set (AZURE_API_KEY/EMBEDDING_HOST)")

    # Avoid running embedding tests when SSL verification is explicitly disabled
    if os.getenv("DISABLE_SSL_VERIFY", "").lower() in ("1", "true", "yes"):
        pytest.skip(
            "Skipping embedding test because DISABLE_SSL_VERIFY is set; may cause TLS failures"
        )

    # Initialize client and run a small embed request
    client = EmbeddingClient()
    vecs = pytest.ensure_temp = client.embed(["hello world"])  # Will be awaited by pytest runner
    # Actually run the coroutine synchronously for testing convenience
    import asyncio

    embeddings = (
        asyncio.get_event_loop().run_until_complete(vecs)
        if asyncio.get_event_loop().is_running()
        else asyncio.run(vecs)
    )
    assert isinstance(embeddings, list)
    assert len(embeddings) == 1
    assert len(embeddings[0]) > 0
