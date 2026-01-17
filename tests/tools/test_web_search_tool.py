import pytest
from unittest.mock import patch, MagicMock
from src.services.search import web_search

class TestWebSearchTool:

    @patch("src.services.search.get_provider")
    def test_search_calls_provider_and_returns_result(self, mock_get_provider):
        # Mock provider object with a .search method that returns a response object
        provider_mock = MagicMock()
        response_mock = MagicMock()
        # Emulate to_dict() behavior used by web_search
        response_mock.to_dict.return_value = {
            "timestamp": "2026-01-01T00:00:00",
            "query": "ai trends",
            "answer": "AI is advancing",
            "citations": [],
            "search_results": [],
            "provider": "mock",
        }
        provider_mock.search.return_value = response_mock
        provider_mock.supports_answer = True
        provider_mock.name = "mock"
        mock_get_provider.return_value = provider_mock

        result = web_search("ai trends")

        assert isinstance(result, dict)
        assert result["answer"] == "AI is advancing"
        mock_get_provider.assert_called_once()

    @patch("src.services.search.get_provider")
    @patch("src.services.search.AnswerConsolidator.consolidate")
    def test_consolidation_called_for_serp(self, mock_consolidate, mock_get_provider):
        # Provider that does not support answer triggers consolidation
        provider_mock = MagicMock()
        response_mock = MagicMock()
        response_mock.to_dict.return_value = {"search_results": [], "provider": "serper", "query": "q"}
        provider_mock.search.return_value = response_mock
        provider_mock.supports_answer = False
        provider_mock.name = "serper"
        mock_get_provider.return_value = provider_mock

        mock_consolidate.return_value = response_mock

        result = web_search("example", consolidation="template")

        # Consolidator should have been invoked
        mock_consolidate.assert_called_once()
        assert isinstance(result, dict)
