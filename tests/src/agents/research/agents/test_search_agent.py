import pytest
from unittest.mock import patch, MagicMock
from src.agents.research.agents.search_agent import SearchAgent

class TestSearchAgent:
    
    @pytest.fixture
    def agent(self):
        return SearchAgent(
            search_client=MagicMock(),
            llm_client=MagicMock(),
            config={"max_results": 10}
        )
    
    def test_initialization(self, agent):
        assert agent is not None
        assert hasattr(agent, 'search_client')
        assert hasattr(agent, 'llm_client')
        assert agent.config["max_results"] == 10
        
    @patch('src.agents.research.agents.search_agent.SearchAgent._perform_search')
    def test_search(self, mock_search, agent):
        query = "latest advancements in AI"
        mock_search.return_value = ["Result 1", "Result 2"]
        
        result = agent.search(query)
        
        assert len(result) == 2
        assert "Result 1" in result
        mock_search.assert_called_once_with(query)
        
    @patch('src.agents.research.agents.search_agent.SearchAgent._refine_query')
    def test_refine_query(self, mock_refine, agent):
        original_query = "AI advancements"
        mock_refine.return_value = "latest breakthroughs in artificial intelligence research"
        
        result = agent.refine_query(original_query)
        
        assert result == "latest breakthroughs in artificial intelligence research"
        mock_refine.assert_called_once_with(original_query)
        
    @patch('src.agents.research.agents.search_agent.SearchAgent._extract_relevant_info')
    def test_extract_relevant_info(self, mock_extract, agent):
        search_results = ["Result 1", "Result 2"]
        mock_extract.return_value = ["Extracted info 1", "Extracted info 2"]
        
        result = agent.extract_relevant_info(search_results)
        
        assert len(result) == 2
        assert "Extracted info 1" in result
        mock_extract.assert_called_once_with(search_results)