# -*- coding: utf-8 -*-
import os
import pytest

from importlib import import_module

cloud = import_module("src.services.llm.cloud_provider")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_openai_complete_integration():
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")

    # Minimal smoke test - ensure complete() returns a string
    res = await cloud.complete(
        prompt="Hello world",
        system_prompt="You are a helpful assistant.",
        model="gpt-3.5-turbo",
    )
    assert isinstance(res, str)
