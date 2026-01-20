# -*- coding: utf-8 -*-
import os

import pytest

from src.services.search.providers.tavily import TavilyProvider
from src.services.search.types import WebSearchResponse


@pytest.mark.asyncio
async def test_tavily_search_real():
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        pytest.skip("TAVILY_API_KEY not set")

    provider = TavilyProvider(api_key=api_key)
    response = provider.search("DeepTutor AI assistant", num_results=5)

    assert isinstance(response, WebSearchResponse)
    assert response.query == "DeepTutor AI assistant"
    assert len(response.search_results) > 0
    assert response.provider == "tavily"
