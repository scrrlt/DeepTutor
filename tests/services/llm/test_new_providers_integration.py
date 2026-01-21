# -*- coding: utf-8 -*-
"""Integration tests for newly added LLM providers."""
import os

import pytest

from src.services.llm import factory


@pytest.mark.asyncio
async def test_groq_llm_real():
    """
    Integration test that verifies a Groq LLM responds with the word "ready".
    
    Reads the `GROQ_API_KEY` environment variable and skips the test if it is not set. Calls the LLM completion factory against Groq's API and asserts the returned text contains "ready" (case-insensitive).
    """
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


@pytest.mark.asyncio
async def test_mistral_llm_real():
    """
    Integration test that verifies the Mistral provider returns a readiness phrase.
    
    Skips the test if MISTRAL_API_KEY is not set. Calls the LLM factory with a prompt asking for "Mistral Ready" using model "mistral-small-latest" and asserts the response contains "ready" (case-insensitive).
    """
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