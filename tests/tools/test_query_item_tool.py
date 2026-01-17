import pytest
from unittest.mock import patch, MagicMock
from src.tools.query_item_tool import QueryItemTool

class TestQueryItemTool:
    
    @pytest.fixture
    def query_item_tool(self):
        return QueryItemTool(
            llm_client=MagicMock(),
            token_tracker=MagicMock()
        )
    
    def test_initialization(self, query_item_tool):
        assert query_item_tool is not None
        assert hasattr(query_item_tool, 'llm_client')
        assert hasattr(query_item_tool, 'token_tracker')
        
    @patch('src.agents.research.agents.reporting_agent.ReportingAgent._generate_report_section')
    def test_generate_executive_summary(self, mock_generate, query_item_tool):
        research_data = MagicMock()
        research_data.notes = ["Note 1", "Note 2"]
        research_data.query = "Test query"
        
        mock_generate.return_value = "Executive Summary"
        result = query_item_tool.generate_executive_summary(research_data)
        
        assert result == "Executive Summary"
        mock_generate.assert_called_once()
        
    @patch('src.agents.research.agents.reporting_agent.ReportingAgent._generate_report_section')
    def test_generate_methodology(self, mock_generate, query_item_tool):
        research_data = MagicMock()
        mock_generate.return_value = "Methodology Section"
        
        result = query_item_tool.generate_methodology(research_data)
        
        assert result == "Methodology Section"
        mock_generate.assert_called_once()
        
    @patch('src.agents.research.agents.reporting_agent.ReportingAgent._generate_report_section')
    def test_generate_findings(self, mock_generate, query_item_tool):
        research_data = MagicMock()
        research_data.notes = ["Finding 1", "Finding 2"]
        
        mock_generate.return_value = "Findings Section"
        result = query_item_tool.generate_findings(research_data)
        
        assert result == "Findings Section"
        mock_generate.assert_called_once()
        
    @patch('src.agents.research.agents.reporting_agent.ReportingAgent._compile_report')
    def test_generate_full_report(self, mock_compile, query_item_tool):
        research_data = MagicMock()
        mock_compile.return_value = "Full Report"
        
        result = query_item_tool.generate_full_report(research_data)
        
        assert result == "Full Report"
        mock_compile.assert_called_once_with(research_data)