import pytest
from unittest.mock import patch, MagicMock
from src.agents.solve.agents.problem_analyzer import ProblemAnalyzer

class TestProblemAnalyzer:
    
    @pytest.fixture
    def analyzer(self):
        return ProblemAnalyzer(
            llm_client=MagicMock(),
            config={"analysis_depth": "detailed"}
        )
    
    def test_initialization(self, analyzer):
        assert analyzer is not None
        assert hasattr(analyzer, 'llm_client')
        assert analyzer.config["analysis_depth"] == "detailed"
        
    # NOTE: Removed invalid incomplete @patch('src decorator detected by static analysis.
