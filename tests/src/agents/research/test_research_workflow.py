import pytest
from unittest.mock import patch, MagicMock
from src.agents.research.research_workflow import ResearchWorkflow

class TestResearchWorkflow:
    
    @pytest.fixture
    def workflow(self):
        return ResearchWorkflow(
            search_agent=MagicMock(),
            extraction_agent=MagicMock(),
            reporting_agent=MagicMock(),
            config={"depth": "comprehensive"}
        )
    
    def test_initialization(self, workflow):
        assert workflow is not None
        assert hasattr(workflow, 'search_agent')
        assert hasattr(workflow, 'extraction_agent')
        assert hasattr(workflow, 'reporting_agent')
        assert workflow.config["depth"] == "comprehensive"
        
    @patch('src.agents.research.research_workflow.ResearchWorkflow._conduct_research')
    def test_conduct_research(self, mock_research, workflow):
        query = "Impact of AI on healthcare"
        mock_research.return_value = {"research_data": "Comprehensive research data"}
        
        result = workflow.conduct_research(query)
        
        assert result["research_data"] == "Comprehensive research data"
        mock_research.assert_called_once_with(query)
        
    @patch('src.agents.research.research_workflow.ResearchWorkflow._search_and_extract')
    def test_search_and_extract(self, mock_search_extract, workflow):
        query = "AI healthcare"
        mock_search_extract.return_value = {
            "search_results": ["Result 1", "Result 2"],
            "extracted_info": ["Info 1", "Info 2"]
        }
        
        result = workflow._search_and_extract(query)
        
        assert "search_results" in result
        assert "extracted_info" in result
        assert len(result["search_results"]) == 2
        mock_search_extract.assert_called_once_with(query)
        
    @patch('src.agents.research.research_workflow.ResearchWorkflow._generate_report')
    def test_generate_report(self, mock_generate, workflow):
        research_data = {"key": "value"}
        mock_generate.return_value = "Comprehensive research report"
        
        result = workflow.generate_report(research_data)
        
        assert result == "Comprehensive research report"
        mock_generate.assert_called_once_with(research_data)