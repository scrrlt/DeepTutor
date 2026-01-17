import pytest
from unittest.mock import patch, MagicMock
from src.agents.solve.agents.solution_generator import SolutionGenerator

class TestSolutionGenerator:
    
    @pytest.fixture
    def generator(self):
        return SolutionGenerator(
            llm_client=MagicMock(),
            config={"creativity": 0.8, "max_solutions": 3}
        )
    
    def test_initialization(self, generator):
        assert generator is not None
        assert hasattr(generator, 'llm_client')
        assert generator.config["creativity"] == 0.8
        assert generator.config["max_solutions"] == 3
        
    @patch('src.agents.solve.agents.solution_generator.SolutionGenerator._generate_solutions')
    def test_generate(self, mock_generate, generator):
        problem = {"description": "Test problem", "constraints": ["Constraint 1"]}
        mock_generate.return_value = ["Solution 1", "Solution 2"]
        
        result = generator.generate(problem)
        
        assert len(result) == 2
        assert "Solution 1" in result
        mock_generate.assert_called_once_with(problem)
        
    @patch('src.agents.solve.agents.solution_generator.SolutionGenerator._rank_solutions')
    def test_rank_solutions(self, mock_rank, generator):
        solutions = ["Solution A", "Solution B", "Solution C"]
        mock_rank.return_value = [
            {"solution": "Solution B", "score": 0.9},
            {"solution": "Solution A", "score": 0.7},
            {"solution": "Solution C", "score": 0.5}
        ]
        
        result = generator.rank_solutions(solutions)
        
        assert len(result) == 3
        assert result[0]["solution"] == "Solution B"
        assert result[0]["score"] == 0.9
        mock_rank.assert_called_once_with(solutions)
        
    @patch('src.agents.solve.agents.solution_generator.SolutionGenerator._elaborate_solution')
    def test_elaborate_solution(self, mock_elaborate, generator):
        solution = "Brief solution"
        mock_elaborate.return_value = "Detailed solution with steps"
        
        result = generator.elaborate_solution(solution)
        
        assert result == "Detailed solution with steps"
        mock_elaborate.assert_called_once_with(solution)