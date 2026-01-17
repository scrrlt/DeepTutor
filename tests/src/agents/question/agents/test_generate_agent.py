import pytest
from unittest.mock import patch, MagicMock
from src.agents.question.agents.generate_agent import GenerateAgent

class TestGenerateAgent:
    
    @pytest.fixture
    def agent(self):
        return GenerateAgent(
            llm_client=MagicMock(),
            config={"max_tokens": 500, "temperature": 0.7}
        )
    
    def test_initialization(self, agent):
        assert agent is not None
        assert hasattr(agent, 'llm_client')
        assert agent.config["max_tokens"] == 500
        
    @patch('src.agents.question.agents.generate_agent.GenerateAgent._generate_response')
    def test_generate(self, mock_generate, agent):
        context = ["Context 1", "Context 2"]
        question = "What is machine learning?"
        mock_generate.return_value = "Machine learning is a subset of AI..."
        
        result = agent.generate(question, context)
        
        assert result == "Machine learning is a subset of AI..."
        mock_generate.assert_called_once_with(question, context)
        
    @patch('src.agents.question.agents.generate_agent.GenerateAgent._create_prompt')
    def test_create_prompt(self, mock_create, agent):
        mock_create.return_value = "Formatted prompt"
        
        result = agent._create_prompt("Question", ["Context"])
        
        assert result == "Formatted prompt"
        mock_create.assert_called_once_with("Question", ["Context"])
        
    @patch('src.agents.question.agents.generate_agent.GenerateAgent._validate_response')
    def test_validate_response(self, mock_validate, agent):
        mock_validate.return_value = True
        
        result = agent._validate_response("Response")
        
        assert result is True
        mock_validate.assert_called_once_with("Response")