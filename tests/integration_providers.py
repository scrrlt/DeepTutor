# -*- coding: utf-8 -*-
"""Integration tests for LLM and Embedding providers."""

import asyncio
import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

# Load environment variables from .env.local if it exists
PROJECT_ROOT = Path(__file__).resolve().parent.parent
env_local = PROJECT_ROOT / ".env.local"
if env_local.exists():
    load_dotenv(env_local, override=True)

from src.services.llm import complete as llm_complete
from src.services.embedding import get_embedding_client


RUN_NETWORK_TESTS = os.getenv("RUN_LLM_NETWORK_TESTS") == "1"

pytestmark = pytest.mark.skipif(
    not RUN_NETWORK_TESTS,
    reason="Set RUN_LLM_NETWORK_TESTS=1 to enable integration provider calls.",
)


@pytest.mark.asyncio
async def test_llm_provider_integration():
    """Verify the active LLM provider works as intended."""
    api_key = os.getenv("LLM_API_KEY")
    if not api_key and "localhost" not in os.getenv("LLM_HOST", ""):
        pytest.skip("LLM_API_KEY not set and not using local host, skipping integration test")

    # Bare call letting pytest report failures naturally
    response = await asyncio.wait_for(
        llm_complete("Say 'DeepTutor' and nothing else."),
        timeout=30.0,
    )
    assert "DeepTutor" in response
    print(f"\u2713 LLM provider verified: {os.getenv('LLM_BINDING', 'openai')}")


@pytest.mark.asyncio
async def test_embedding_provider_integration():
    """Verify the active Embedding provider works as intended."""
    api_key = os.getenv("EMBEDDING_API_KEY")
    if not api_key and "localhost" not in os.getenv("EMBEDDING_HOST", ""):
        pytest.skip("EMBEDDING_API_KEY not set and not using local host, skipping integration test")

    # Bare call letting pytest report failures naturally
    client = get_embedding_client()
    embeddings = await client.embed(["DeepTutor is awesome"])
    assert len(embeddings) == 1
    assert len(embeddings[0]) > 0
    print(f"\u2713 Embedding provider verified: {os.getenv('EMBEDDING_BINDING', 'openai')}")
