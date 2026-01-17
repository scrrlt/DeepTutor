import pytest
from unittest.mock import patch, MagicMock
from src.agents.solve.agents.verification_agent import VerificationAgent

class TestVerificationAgent:
    
    @pytest.fixture
    def agent(self):
        return VerificationAgent(
            llm_client=MagicMock(),
            config={"verification_criteria": ["correctness", "efficiency"]}
        )
    
    def test_initialization(self, agent):
        assert agent is not None
        assert hasattr(agent, 'llm_client')
        assert "correctness" in agent.config["verification_criteria"]
        
    @patch('src.agents.solve.agents.verification_agent.VerificationAgent._verify_solution')
    def test_verify(self, mock_verify, agent):
        problem = {"description": "Problem description"}
        solution = "Proposed solution"
        mock_verify.return_value = {
            "is_valid": True,
            "score": 0.85,
            "feedback": "Good solution"
        }
        
        result = agent.verify(problem, solution)
        
        assert result["is_valid"] is True
        assert result["score"] == 0.85
        assert result["feedback"] == "Good solution"
        mock_verify.assert_called_once_with(problem, solution)
        
    @patch('src.agents.solve.agents.verification_agent.VerificationAgent._check_correctness')
    def test_check_correctness(self, mock_check, agent):
        problem = {"description": "Problem description"}
        solution = "Proposed solution"
        mock_check.return_value = {"is_correct": True, "explanation": "Solution is correct"}
        
        result = agent.check_correctness(problem, solution)
        
        assert result["is_correct"] is True
        assert result["explanation"] == "Solution is correct"
        mock_check.assert_called_once_with(problem, solution)
        
    @patch('src.agents.solve.agents.verification_agent.VerificationAgent._provide_improvement_suggestions')
    def test_provide_improvement_suggestions(self, mock_suggestions, agent):
        solution = "Proposed solution"
        verification_result = {"is_valid": True, "weak_points": ["Point 1"]}
        mock_suggestions.return_value = ["Suggestion 1", "Suggestion 2"]
        
        result = agent.provide_improvement_suggestions(solution, verification_result)
        
        assert len(result) == 2
        assert "Suggestion 1" in result
        mock_suggestions.assert_called_once_with(solution, verification_result)