import os

import pytest

from src.services.search.providers.serper import SerperProvider
from src.services.search.types import WebSearchResponse


@pytest.mark.asyncio
async def test_serper_search_real():
    api_key = os.getenv("SERPER_API_KEY")
    if not api_key:
        pytest.skip("SERPER_API_KEY not set")

    provider = SerperProvider(api_key=api_key)
    response = await provider.search("DeepTutor AI assistant", num=5)

    assert isinstance(response, WebSearchResponse)
    assert response.query == "DeepTutor AI assistant"
    assert len(response.search_results) > 0
    assert response.provider == "serper"
