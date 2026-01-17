import pytest
from unittest.mock import patch, MagicMock
from src.agents.solve.solver_workflow import SolverWorkflow

class TestSolverWorkflow:
    
    @pytest.fixture
    def workflow(self):
        return SolverWorkflow(
            problem_analyzer=MagicMock(),
            solution_generator=MagicMock(),
            verification_agent=MagicMock(),
            config={"max_iterations": 3}
        )
    
    def test_initialization(self, workflow):
        assert workflow is not None
        assert hasattr(workflow, 'problem_analyzer')
        assert hasattr(workflow, 'solution_generator')
        assert hasattr(workflow, 'verification_agent')
        assert workflow.config["max_iterations"] == 3
        
    @patch('src.agents.solve.solver_workflow.SolverWorkflow._solve_problem')
    def test_solve(self, mock_solve, workflow):
        problem_statement = "How to optimize database queries?"
        mock_solve.return_value = {
            "solution": "Detailed solution",
            "verification_result": {"is_valid": True, "score": 0.9}
        }
        
        result = workflow.solve(problem_statement)
        
        assert result["solution"] == "Detailed solution"
        assert result["verification_result"]["is_valid"] is True
        mock_solve.assert_called_once_with(problem_statement)
        
    @patch('src.agents.solve.solver_workflow.SolverWorkflow._analyze_problem')
    def test_analyze_problem(self, mock_analyze, workflow):
        problem_statement = "How to optimize database queries?"
        mock_analyze.return_value = {
            "problem_type": "optimization",
            "key_constraints": ["Performance", "Scalability"]
        }
        
        result = workflow._analyze_problem(problem_statement)
        
        assert result["problem_type"] == "optimization"
        assert "Performance" in result["key_constraints"]
        mock_analyze.assert_called_once_with(problem_statement)
        
    @patch('src.agents.solve.solver_workflow.SolverWorkflow._generate_and_verify')
    def test_generate_and_verify(self, mock_gen_verify, workflow):
        problem_analysis = {"problem_type": "optimization"}
        mock_gen_verify.return_value = {
            "solution": "Optimized solution",
            "is_valid": True
        }
        
        result = workflow._generate_and_verify(problem_analysis)
        
        assert result["solution"] == "Optimized solution"
        assert result["is_valid"] is True
        mock_gen_verify.assert_called_once_with(problem_analysis)