import pytest
from unittest.mock import patch, MagicMock
from src.tools.knowledge_base_tool import KnowledgeBaseTool

class TestKnowledgeBaseTool:
    
    @pytest.fixture
    def tool(self):
        return KnowledgeBaseTool(
            vector_store=MagicMock(),
            config={"top_k": 5, "similarity_threshold": 0.7}
        )
    
    def test_initialization(self, tool):
        assert tool is not None
        assert hasattr(tool, 'vector_store')
        assert tool.config["top_k"] == 5
        assert tool.config["similarity_threshold"] == 0.7
        
    @patch('src.tools.knowledge_base_tool.KnowledgeBaseTool._query_knowledge_base')
    def test_query(self, mock_query, tool):
        query = "How do neural networks work?"
        mock_query.return_value = [
            {"text": "Neural networks are...", "score": 0.85},
            {"text": "Deep learning models...", "score": 0.78}
        ]
        
        result = tool.query(query)
        
        assert len(result) == 2
        assert result[0]["score"] == 0.85
        assert "Neural networks are..." in result[0]["text"]
        mock_query.assert_called_once_with(query)
        
    @patch('src.tools.knowledge_base_tool.KnowledgeBaseTool._filter_by_threshold')
    def test_filter_by_threshold(self, mock_filter, tool):
        results = [
            {"text": "Relevant text", "score": 0.8},
            {"text": "Less relevant", "score": 0.6}
        ]
        mock_filter.return_value = [{"text": "Relevant text", "score": 0.8}]
        
        result = tool.filter_results(results)
        
        assert len(result) == 1
        assert result[0]["text"] == "Relevant text"
        mock_filter.assert_called_once_with(results)
        
    @patch('src.tools.knowledge_base_tool.KnowledgeBaseTool._format_results')
    def test_format_results(self, mock_format, tool):
        results = [{"text": "Text 1", "score": 0.8}, {"text": "Text 2", "score": 0.75}]
        mock_format.return_value = ["Formatted text 1", "Formatted text 2"]
        
        result = tool._format_results(results)
        
        assert len(result) == 2
        assert "Formatted text 1" in result
        mock_format.assert_called_once_with(results)