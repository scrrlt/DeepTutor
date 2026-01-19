# -*- coding: utf-8 -*-
import os

import pytest

from src.services.llm import factory


RUN_NETWORK_TESTS = os.getenv("RUN_LLM_NETWORK_TESTS") == "1"

pytestmark = pytest.mark.skipif(
    not RUN_NETWORK_TESTS,
    reason="Set RUN_LLM_NETWORK_TESTS=1 to enable real provider calls.",
)


@pytest.mark.asyncio
async def test_openai_llm_real():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set")

    response = await factory.complete(
        prompt="Hello, say 'OpenAI Ready'",
        model="gpt-4o",
        api_key=api_key,
        base_url="https://api.openai.com/v1",
    )
    assert "OpenAI Ready" in response


@pytest.mark.asyncio
async def test_gemini_llm_real():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        pytest.skip("GEMINI_API_KEY not set")

    base_url = os.getenv("GEMINI_BASE_URL")
    if not base_url:
        pytest.skip(
            "Gemini integration requires an OpenAI-compatible base URL via "
            "GEMINI_BASE_URL."
        )
    if "generativelanguage.googleapis.com" in base_url:
        pytest.skip(
            "Gemini requires an OpenAI-compatible gateway; "
            "set GEMINI_BASE_URL accordingly."
        )

    response = await factory.complete(
        prompt="Hello, say 'Gemini Ready'",
        model="gemini-1.5-pro",
        api_key=api_key,
        base_url=base_url,
    )
    assert "Gemini Ready" in response


@pytest.mark.asyncio
async def test_claude_llm_real():
    api_key = os.getenv("CLAUDE_API_KEY")
    if not api_key:
        pytest.skip("CLAUDE_API_KEY not set")

    response = await factory.complete(
        prompt="Hello, say 'Claude Ready'",
        model="claude-3-5-sonnet-20241022",
        api_key=api_key,
        base_url="https://api.anthropic.com/v1",
        binding="anthropic",
    )
    assert "Claude Ready" in response
