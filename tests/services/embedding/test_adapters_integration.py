import os

import pytest

from src.services.embedding.adapters.base import EmbeddingRequest
from src.services.embedding.adapters.cohere import CohereEmbeddingAdapter
from src.services.embedding.adapters.jina import JinaEmbeddingAdapter
from src.services.embedding.adapters.openai_compatible import (
    OpenAICompatibleEmbeddingAdapter,
)

RUN_NETWORK_TESTS = os.getenv("RUN_LLM_NETWORK_TESTS") == "1"

pytestmark = pytest.mark.skipif(
    not RUN_NETWORK_TESTS,
    reason="Set RUN_LLM_NETWORK_TESTS=1 to enable real embedding calls.",
)


@pytest.mark.asyncio
async def test_openai_adapter_real():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set")

    config = {
        "api_key": api_key,
        "base_url": "https://api.openai.com/v1",
        "model": "text-embedding-3-small",
    }
    adapter = OpenAICompatibleEmbeddingAdapter(config)
    request = EmbeddingRequest(texts=["DeepTutor test case"], model="text-embedding-3-small")

    response = await adapter.embed(request)
    assert len(response.embeddings) == 1
    assert len(response.embeddings[0]) == 1536


@pytest.mark.asyncio
async def test_jina_adapter_real():
    api_key = os.getenv("JINA_API_KEY")
    if not api_key:
        pytest.skip("JINA_API_KEY not set")

    config = {
        "api_key": api_key,
        "base_url": "https://api.jina.ai/v1",
        "model": "jina-embeddings-v3",
    }
    adapter = JinaEmbeddingAdapter(config)
    request = EmbeddingRequest(texts=["DeepTutor test case"], model="jina-embeddings-v3")

    response = await adapter.embed(request)
    assert len(response.embeddings) == 1
    assert len(response.embeddings[0]) == 1024


@pytest.mark.asyncio
async def test_cohere_adapter_real():
    api_key = os.getenv("COHERE_API_KEY")
    if not api_key:
        pytest.skip("COHERE_API_KEY not set")

    config = {
        "api_key": api_key,
        "base_url": "https://api.cohere.ai",
        "model": "embed-english-v3.0",
    }
    adapter = CohereEmbeddingAdapter(config)
    request = EmbeddingRequest(
        texts=["DeepTutor test case"],
        model="embed-english-v3.0",
        input_type="search_document",
    )

    response = await adapter.embed(request)
    assert len(response.embeddings) == 1
    assert len(response.embeddings[0]) == 1024
