import pytest
from unittest.mock import patch, MagicMock
from src.agents.research.agents.extraction_agent import ExtractionAgent

class TestExtractionAgent:
    
    @pytest.fixture
    def agent(self):
        return ExtractionAgent(
            llm_client=MagicMock(),
            config={"extraction_mode": "detailed"}
        )
    
    def test_initialization(self, agent):
        assert agent is not None
        assert hasattr(agent, 'llm_client')
        assert agent.config["extraction_mode"] == "detailed"
        
    @patch('src.agents.research.agents.extraction_agent.ExtractionAgent._extract_information')
    def test_extract(self, mock_extract, agent):
        text = "Sample research text with important information."
        mock_extract.return_value = {
            "key_points": ["Point 1", "Point 2"],
            "entities": ["Entity 1", "Entity 2"]
        }
        
        result = agent.extract(text)
        
        assert "key_points" in result
        assert len(result["key_points"]) == 2
        assert "entities" in result
        mock_extract.assert_called_once_with(text)
        
    @patch('src.agents.research.agents.extraction_agent.ExtractionAgent._identify_key_points')
    def test_identify_key_points(self, mock_identify, agent):
        text = "Research text with key points."
        mock_identify.return_value = ["Point 1", "Point 2"]
        
        result = agent.identify_key_points(text)
        
        assert len(result) == 2
        assert "Point 1" in result
        mock_identify.assert_called_once_with(text)
        
    @patch('src.agents.research.agents.extraction_agent.ExtractionAgent._extract_citations')
    def test_extract_citations(self, mock_extract_citations, agent):
        text = "Text with citations (Smith, 2020)."
        mock_extract_citations.return_value = [{"author": "Smith", "year": 2020}]
        
        result = agent.extract_citations(text)
        
        assert len(result) == 1
        assert result[0]["author"] == "Smith"
        assert result[0]["year"] == 2020
        mock_extract_citations.assert_called_once_with(text)