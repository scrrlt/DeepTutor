import pytest
from unittest.mock import patch, MagicMock
from src.tools.text_analysis_tool import TextAnalysisTool

class TestTextAnalysisTool:
    
    @pytest.fixture
    def tool(self):
        return TextAnalysisTool(
            nlp_engine=MagicMock(),
            config={"analysis_types": ["sentiment", "entities", "summary"]}
        )
    
    def test_initialization(self, tool):
        assert tool is not None
        assert hasattr(tool, 'nlp_engine')
        assert "sentiment" in tool.config["analysis_types"]
        
    @patch('src.tools.text_analysis_tool.TextAnalysisTool._analyze_text')
    def test_analyze(self, mock_analyze, tool):
        text = "The product is excellent and user-friendly. John Smith from Acme Corp loved it."
        mock_analyze.return_value = {
            "sentiment": {"positive": 0.8, "neutral": 0.15, "negative": 0.05},
            "entities": [{"text": "John Smith", "type": "PERSON"}, {"text": "Acme Corp", "type": "ORG"}],
            "summary": "Positive review of a user-friendly product by John Smith from Acme Corp."
        }
        
        result = tool.analyze(text)
        
        assert "sentiment" in result
        assert result["sentiment"]["positive"] == 0.8
        assert len(result["entities"]) == 2
        mock_analyze.assert_called_once_with(text)
        
    @patch('src.tools.text_analysis_tool.TextAnalysisTool._analyze_sentiment')
    def test_analyze_sentiment(self, mock_sentiment, tool):
        text = "The product is excellent."
        mock_sentiment.return_value = {"positive": 0.9, "neutral": 0.1, "negative": 0.0}
        
        result = tool.analyze_sentiment(text)
        
        assert result["positive"] == 0.9
        mock_sentiment.assert_called_once_with(text)
        
    @patch('src.tools.text_analysis_tool.TextAnalysisTool._extract_entities')
    def test_extract_entities(self, mock_extract, tool):
        text = "John Smith works at Acme Corp."
        mock_extract.return_value = [
            {"text": "John Smith", "type": "PERSON"},
            {"text": "Acme Corp", "type": "ORG"}
        ]
        
        result = tool.extract_entities(text)
        
        assert len(result) == 2
        assert result[0]["text"] == "John Smith"
        assert result[1]["type"] == "ORG"
        mock_extract.assert_called_once_with(text)