import pytest
from unittest.mock import patch, MagicMock
from src.agents.question.agents.retrieve_agent import RetrieveAgent

class TestRetrieveAgent:
    
    @pytest.fixture
    def agent(self):
        return RetrieveAgent(
            knowledge_base=MagicMock(),
            config={"top_k": 5}
        )
    
    def test_initialization(self, agent):
        assert agent is not None
        assert hasattr(agent, 'knowledge_base')
        assert agent.config["top_k"] == 5
        
    @patch('src.agents.question.agents.retrieve_agent.RetrieveAgent._retrieve_contexts')
    def test_retrieve(self, mock_retrieve, agent):
        question = "What is reinforcement learning?"
        mock_retrieve.return_value = ["Context 1", "Context 2"]
        
        result = agent.retrieve(question)
        
        assert len(result) == 2
        assert "Context 1" in result
        mock_retrieve.assert_called_once_with(question)
        
    @patch('src.agents.question.agents.retrieve_agent.RetrieveAgent._preprocess_query')
    def test_preprocess_query(self, mock_preprocess, agent):
        mock_preprocess.return_value = "processed query"
        
        result = agent._preprocess_query("raw query")
        
        assert result == "processed query"
        mock_preprocess.assert_called_once_with("raw query")
        
    @patch('src.agents.question.agents.retrieve_agent.RetrieveAgent._format_contexts')
    def test_format_contexts(self, mock_format, agent):
        contexts = [{"text": "Context 1"}, {"text": "Context 2"}]
        mock_format.return_value = ["Formatted context 1", "Formatted context 2"]
        
        result = agent._format_contexts(contexts)
        
        assert len(result) == 2
        assert "Formatted context 1" in result
        mock_format.assert_called_once_with(contexts)