# -*- coding: utf-8 -*-
import os
import pytest
import importlib

from src.services.search import get_provider


@pytest.mark.integration
def test_perplexity_search_integration():
    api_key = os.getenv("PERPLEXITY_API_KEY")
    if not api_key:
        pytest.skip("PERPLEXITY_API_KEY not set")

    provider = get_provider("perplexity", api_key=api_key)
    resp = provider.search("What is AI?")
    assert resp.provider == "perplexity"
    assert isinstance(resp.query, str)
