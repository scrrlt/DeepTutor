import pytest
from unittest.mock import patch, MagicMock
from src.tools.web_search_tool import WebSearchTool

class TestWebSearchTool:
    
    @pytest.fixture
    def tool(self):
        return WebSearchTool(
            search_api_client=MagicMock(),
            config={"max_results": 5, "safe_search": True}
        )
    
    def test_initialization(self, tool):
        assert tool is not None
        assert hasattr(tool, 'search_api_client')
        assert tool.config["max_results"] == 5
        assert tool.config["safe_search"] is True
        
    @patch('src.tools.web_search_tool.WebSearchTool._execute_search')
    def test_search(self, mock_search, tool):
        query = "artificial intelligence trends"
        mock_search.return_value = ["Result 1", "Result 2"]
        
        result = tool.search(query)
        
        assert len(result) == 2
        assert "Result 1" in result
        mock_search.assert_called_once_with(query)
        
    @patch('src.tools.web_search_tool.WebSearchTool._format_results')
    def test_format_results(self, mock_format, tool):
        raw_results = [{"title": "Title 1", "snippet": "Snippet 1", "link": "http://example.com"}]
        mock_format.return_value = ["Formatted result 1"]
        
        result = tool._format_results(raw_results)
        
        assert len(result) == 1
        assert "Formatted result 1" in result
        mock_format.assert_called_once_with(raw_results)
        
    @patch('src.tools.web_search_tool.WebSearchTool._filter_results')
    def test_filter_results(self, mock_filter, tool):
        results = ["Result 1", "Result 2", "Result 3"]
        mock_filter.return_value = ["Result 1", "Result 3"]
        
        result = tool.filter_results(results, "filter criteria")
        
        assert len(result) == 2
        assert "Result 2" not in result
        mock_filter.assert_called_once_with(results, "filter criteria")