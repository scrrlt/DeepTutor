import pytest
from unittest.mock import patch, MagicMock
from src.agents.question.agents.relevance_analyzer import RelevanceAnalyzer

class TestRelevanceAnalyzer:
    
    @pytest.fixture
    def analyzer(self):
        return RelevanceAnalyzer(
            llm_client=MagicMock(),
            config={"threshold": 0.7}
        )
    
    def test_initialization(self, analyzer):
        assert analyzer is not None
        assert analyzer.config["threshold"] == 0.7
        
    @patch('src.agents.question.agents.relevance_analyzer.RelevanceAnalyzer._analyze_relevance')
    def test_analyze(self, mock_analyze, analyzer):
        question = "What is deep learning?"
        context = ["Context about deep learning", "Unrelated context"]
        mock_analyze.return_value = [
            {"text": "Context about deep learning", "score": 0.9},
            {"text": "Unrelated context", "score": 0.2}
        ]
        
        result = analyzer.analyze(question, context)
        
        assert len(result) == 2
        assert result[0]["score"] == 0.9
        mock_analyze.assert_called_once_with(question, context)
        
    @patch('src.agents.question.agents.relevance_analyzer.RelevanceAnalyzer._filter_by_threshold')
    def test_filter_by_threshold(self, mock_filter, analyzer):
        scored_contexts = [
            {"text": "Relevant", "score": 0.8},
            {"text": "Not relevant", "score": 0.3}
        ]
        mock_filter.return_value = [{"text": "Relevant", "score": 0.8}]
        
        result = analyzer.filter_relevant_contexts(scored_contexts)
        
        assert len(result) == 1
        assert result[0]["text"] == "Relevant"
        mock_filter.assert_called_once_with(scored_contexts)