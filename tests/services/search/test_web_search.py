#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests for the web_search function.
"""

import pytest
from unittest.mock import patch, MagicMock, ANY

from src.services.search import web_search
from src.services.search.types import WebSearchResponse


@pytest.mark.parametrize("provider_name", ["perplexity", "baidu", "tavily", "exa", "serper", "jina"])
@patch('src.services.search.get_provider')
def test_web_search_with_different_providers(mock_get_provider, provider_name):
    """
    Tests that the web_search function can be called with different providers.
    """
    mock_provider = MagicMock()
    mock_provider.name = provider_name
    mock_provider.search.return_value = WebSearchResponse(query="test query")
    mock_get_provider.return_value = mock_provider

    web_search("test query", provider=provider_name)

    mock_get_provider.assert_called_with(provider_name)
    if provider_name == "serper":
        mock_provider.search.assert_called_with("test query", num=10)
    else:
        mock_provider.search.assert_called_with("test query")


@patch('src.services.search.get_provider')
def test_web_search_passes_args_to_provider(mock_get_provider):
    """
    Tests that the web_search function correctly passes the arguments to the provider's search method.
    """
    mock_provider = MagicMock()
    mock_provider.name = "test_provider"
    mock_provider.search.return_value = WebSearchResponse(query="test query")
    mock_get_provider.return_value = mock_provider

    web_search("test query", provider="test_provider", foo="bar", baz=123)

    mock_provider.search.assert_called_with("test query", foo="bar", baz=123)


@patch('src.services.search.get_provider')
@patch('src.services.search._save_results')
def test_web_search_saves_results_to_output_dir(mock_save_results, mock_get_provider):
    """
    Tests that the web_search function correctly handles the output_dir argument and saves the results to a file.
    """
    mock_provider = MagicMock()
    mock_provider.name = "test_provider"
    mock_provider.search.return_value = WebSearchResponse(query="test query")
    mock_get_provider.return_value = mock_provider

    web_search("test query", provider="test_provider", output_dir="/tmp/test_output")

    mock_save_results.assert_called_once_with(ANY, "/tmp/test_output", "test_provider")


@patch('src.services.search.get_provider')
@patch('src.services.search.AnswerConsolidator')
def test_web_search_consolidates_results_for_serp_providers(mock_answer_consolidator, mock_get_provider):
    """
    Tests that the web_search function correctly handles the consolidation argument and consolidates the results for SERP providers.
    """
    mock_provider = MagicMock()
    mock_provider.name = "serper"
    mock_provider.supports_answer = False
    mock_provider.search.return_value = WebSearchResponse(query="test query")
    mock_get_provider.return_value = mock_provider

    mock_consolidator_instance = MagicMock()
    mock_answer_consolidator.return_value = mock_consolidator_instance

    web_search("test query", provider="serper", consolidation="llm")

    mock_answer_consolidator.assert_called_once_with(consolidation_type='llm', custom_template=None, llm_config=None)
    mock_consolidator_instance.consolidate.assert_called_once()


@patch('src.services.search.get_provider')
def test_web_search_provider_exception(mock_get_provider):
    """
    Tests that the web_search function correctly handles exceptions raised by the provider.
    """
    mock_provider = MagicMock()
    mock_provider.name = "test_provider"
    mock_provider.search.side_effect = Exception("Test Exception")
    mock_get_provider.return_value = mock_provider

    with pytest.raises(Exception, match="Test Exception"):
        web_search("test query", provider="test_provider")