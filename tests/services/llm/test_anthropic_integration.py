# -*- coding: utf-8 -*-
import os
import pytest
import importlib

cloud = importlib.import_module("src.services.llm.cloud_provider")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_anthropic_complete_integration():
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY not set")

    # Minimal smoke test - ensure Anthropic complete returns a string
    res = await cloud.complete(
        prompt="Hello",
        system_prompt="You are a helpful assistant.",
        model="claude-2",  # may be overridden by provider
        api_key=api_key,
        base_url=os.getenv("ANTHROPIC_API_URL"),
        binding="anthropic",
    )

    assert isinstance(res, str)
