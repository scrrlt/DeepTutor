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
    assert "ready" in response.lower()


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
    assert "ready" in response.lower()


@pytest.mark.asyncio
async def test_claude_llm_real():
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY not set")

    response = await factory.complete(
        prompt="Hello, say 'Claude Ready'",
        model="claude-3-5-sonnet-20241022",
        api_key=api_key,
        base_url="https://api.anthropic.com/v1",
        binding="anthropic",
    )
    assert "ready" in response.lower()


@pytest.mark.asyncio
async def test_claude_legacy_key_compat():
    """Test backward compatibility with CLAUDE_API_KEY environment variable."""
    api_key = os.getenv("CLAUDE_API_KEY")
    if not api_key:
        pytest.skip("CLAUDE_API_KEY not set (backward compatibility test)")

    response = await factory.complete(
        prompt="Hello, say 'Claude Ready'",
        model="claude-3-5-sonnet-20241022",
        api_key=api_key,
        base_url="https://api.anthropic.com/v1",
        binding="anthropic",
    )
    assert "ready" in response.lower()


@pytest.mark.asyncio
async def test_together_llm_real():
    api_key = os.getenv("TOGETHER_API_KEY")
    if not api_key:
        pytest.skip("TOGETHER_API_KEY not set")

    response = await factory.complete(
        prompt="Hello, say 'Together Ready'",
        model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        api_key=api_key,
        base_url="https://api.together.xyz/v1",
        binding="together",
    )
    assert "ready" in response.lower()


@pytest.mark.asyncio
async def test_deepseek_llm_real():
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        pytest.skip("DEEPSEEK_API_KEY not set")

    response = await factory.complete(
        prompt="Hello, say 'DeepSeek Ready'",
        model="deepseek-chat",
        api_key=api_key,
        base_url="https://api.deepseek.com",
        binding="deepseek",
    )
    assert "ready" in response.lower()


@pytest.mark.asyncio
async def test_openrouter_llm_real():
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        pytest.skip("OPENROUTER_API_KEY not set")

    response = await factory.complete(
        prompt="Hello, say 'OpenRouter Ready'",
        model="anthropic/claude-3-haiku",
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        binding="openrouter",
    )
    assert "ready" in response.lower()


@pytest.mark.asyncio
async def test_perplexity_llm_real():
    api_key = os.getenv("PERPLEXITY_API_KEY")
    if not api_key:
        pytest.skip("PERPLEXITY_API_KEY not set")

    response = await factory.complete(
        prompt="",
        model="sonar-pro",
        api_key=api_key,
        base_url="https://api.perplexity.ai/chat/completions",
        binding="perplexity",
        messages=[{"role": "user", "content": "Hello, say 'Perplexity Ready'"}],
    )
    assert "ready" in response.lower()
