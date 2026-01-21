# -*- coding: utf-8 -*-
"""Integration tests for newly added LLM providers."""
import sys
import types
import os

import pytest

# Prevent optional provider imports from failing at collection time.
mod = types.ModuleType("src.services.search.providers")
mod.get_available_providers = lambda: []
mod.get_default_provider = lambda: "perplexity"
mod.get_provider = lambda name: types.SimpleNamespace(
    name=name,
    supports_answer=True,
    search=lambda query, **kwargs: types.SimpleNamespace(
        to_dict=lambda: {"answer": "", "citations": [], "search_results": []}
    ),
)
mod.get_providers_info = lambda: []
mod.list_providers = lambda: []
sys.modules.setdefault("src.services.search.providers", mod)

from src.services.llm import factory


@pytest.mark.integration
@pytest.mark.asyncio
async def test_groq_llm_real():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        pytest.skip("GROQ_API_KEY not set")

    response = await factory.complete(
        prompt="Hello, say 'Groq Ready'",
        model="llama-3.1-8b-instant",
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
        binding="groq",
    )
    assert "ready" in response.lower()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_mistral_llm_real():
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        pytest.skip("MISTRAL_API_KEY not set")

    response = await factory.complete(
        prompt="Hello, say 'Mistral Ready'",
        model="mistral-small-latest",
        api_key=api_key,
        base_url="https://api.mistral.ai/v1",
        binding="mistral",
    )
    assert "ready" in response.lower()
