# -*- coding: utf-8 -*-
"""Integration tests for newly added search providers."""
import os

import pytest

from src.services.search.providers.exa import ExaProvider
from src.services.search.types import WebSearchResponse


@pytest.mark.asyncio
async def test_exa_search_real():
    api_key = os.getenv("EXA_API_KEY")
    if not api_key:
        pytest.skip("EXA_API_KEY not set")

    provider = ExaProvider(api_key=api_key)
    response = await provider.search("DeepTutor AI assistant", num_results=5)

    assert isinstance(response, WebSearchResponse)
    assert response.query == "DeepTutor AI assistant"
    assert len(response.search_results) > 0
    assert response.provider == "exa"
