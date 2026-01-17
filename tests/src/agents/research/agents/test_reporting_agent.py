import pytest
from unittest.mock import patch, MagicMock
from src.agents.research.agents.reporting_agent import ReportingAgent

class TestReportingAgent:
    
    @pytest.fixture
    def agent(self):
        return ReportingAgent(
            llm_client=MagicMock(),
            token_tracker=MagicMock()
        )
    
    def test_initialization(self, agent):
        assert agent is not None
        assert hasattr(agent, 'llm_client')
        assert hasattr(agent, 'token_tracker')
        
    @patch('src.agents.research.agents.reporting_agent.ReportingAgent._generate_report_section')
    def test_generate_executive_summary(self, mock_generate, agent):
        research_data = MagicMock()
        research_data.notes = ["Note 1", "Note 2"]
        research_data.query = "Test query"
        
        mock_generate.return_value = "Executive Summary"
        result = agent.generate_executive_summary(research_data)
        
        assert result == "Executive Summary"
        mock_generate.assert_called_once()
        
    @patch('src.agents.research.agents.reporting_agent.ReportingAgent._generate_report_section')
    def test_generate_methodology(self, mock_generate, agent):
        research_data = MagicMock()
        mock_generate.return_value = "Methodology Section"
        
        result = agent.generate_methodology(research_data)
        
        assert result == "Methodology Section"
        mock_generate.assert_called_once()
        
    @patch('src.agents.research.agents.reporting_agent.ReportingAgent._generate_report_section')
    def test_generate_findings(self, mock_generate, agent):
        research_data = MagicMock()
        research_data.notes = ["Finding 1", "Finding 2"]
        
        mock_generate.return_value = "Findings Section"
        result = agent.generate_findings(research_data)
        
        assert result == "Findings Section"
        mock_generate.assert_called_once()
        
    @patch('src.agents.research.agents.reporting_agent.ReportingAgent._compile_report')
    def test_generate_full_report(self, mock_compile, agent):
        research_data = MagicMock()
        mock_compile.return_value = "Full Report"
        
        result = agent.generate_full_report(research_data)
        
        assert result == "Full Report"
        mock_compile.assert_called_once_with(research_data)