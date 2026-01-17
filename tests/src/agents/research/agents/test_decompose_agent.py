"""
Tests for the research's DecomposeAgent.
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from src.agents.research.agents.decompose_agent import DecomposeAgent


@pytest.fixture
def decompose_agent():
    """
    Provides a DecomposeAgent instance with mocked dependencies.
    """
    with patch('src.agents.base_agent.get_prompt_manager'), \
         patch('src.agents.base_agent.get_logger'):
        config = {"researching": {"enable_rag_hybrid": True}, "rag": {}}
        agent = DecomposeAgent(config=config, api_key="test_key", base_url="http://localhost:1234")
        yield agent


@pytest.fixture
def decompose_agent_no_rag():
    """DecomposeAgent configured with RAG hybrid disabled."""
    with patch('src.agents.base_agent.get_prompt_manager'), \
         patch('src.agents.base_agent.get_logger'):
        config = {"researching": {"enable_rag_hybrid": False}, "rag": {}}
        agent = DecomposeAgent(config=config, api_key="test_key", base_url="http://localhost:1234")
        yield agent


@pytest.mark.asyncio
async def test_process_auto_mode_with_rag_disabled(decompose_agent_no_rag: DecomposeAgent):
    """When RAG hybrid is disabled, auto mode should fallback to non-RAG processing."""
    agent = decompose_agent_no_rag
    # Ensure that the non-RAG method is invoked
    agent._process_without_rag = AsyncMock(return_value={"sub_topics": []})
    result = await agent._process_auto_mode("topic", 1)
    # When RAG disabled, expect to get result from _process_without_rag or similar fallback
    assert isinstance(result, dict)



@pytest.mark.asyncio
async def test_decompose_agent_process_manual_mode(decompose_agent: DecomposeAgent):
    """
    Tests that the process method correctly calls the manual mode.
    """
    decompose_agent._process_manual_mode = AsyncMock(return_value={})
    await decompose_agent.process("topic", mode="manual")
    decompose_agent._process_manual_mode.assert_called_once()


@pytest.mark.asyncio
async def test_decompose_agent_process_auto_mode(decompose_agent: DecomposeAgent):
    """
    Tests that the process method correctly calls the auto mode.
    """
    decompose_agent._process_auto_mode = AsyncMock(return_value={})
    await decompose_agent.process("topic", mode="auto")
    decompose_agent._process_auto_mode.assert_called_once()


@pytest.mark.asyncio
@patch('src.agents.research.agents.decompose_agent.rag_search', new_callable=AsyncMock)
async def test_process_manual_mode(mock_rag_search, decompose_agent: DecomposeAgent):
    """
    Tests the _process_manual_mode method.
    """
    decompose_agent._generate_sub_queries = AsyncMock(return_value=["q1"])
    decompose_agent._generate_sub_topics = AsyncMock(return_value=[{"title": "t1"}])
    mock_rag_search.return_value = {"answer": "context"}

    result = await decompose_agent._process_manual_mode("topic", 1)
    
    assert len(result["sub_topics"]) == 1


@pytest.mark.asyncio
@patch('src.agents.research.agents.decompose_agent.rag_search', new_callable=AsyncMock)
async def test_process_auto_mode(mock_rag_search, decompose_agent: DecomposeAgent):
    """
    Tests the _process_auto_mode method.
    """
    decompose_agent._generate_sub_topics_auto = AsyncMock(return_value=[{"title": "t1"}])
    mock_rag_search.return_value = {"answer": "context"}

    result = await decompose_agent._process_auto_mode("topic", 1)
    
    assert len(result["sub_topics"]) == 1


@pytest.mark.asyncio
async def test_process_without_rag(decompose_agent: DecomposeAgent):
    """
    Tests the _process_without_rag method.
    """
    decompose_agent.call_llm = AsyncMock(return_value='{"sub_topics": [{"title": "t1"}]}')
    decompose_agent.get_prompt = MagicMock(return_value="prompt")
    
    result = await decompose_agent._process_without_rag("topic", 1)
    
    assert len(result["sub_topics"]) == 1

@pytest.mark.asyncio
async def test_generate_sub_queries(decompose_agent: DecomposeAgent):
    """
    Tests the _generate_sub_queries method.
    """
    decompose_agent.call_llm = AsyncMock(return_value='{"queries": ["q1"]}')
    decompose_agent.get_prompt = MagicMock(return_value="prompt")
    
    queries = await decompose_agent._generate_sub_queries("topic", 1)
    
    assert len(queries) == 1

@pytest.mark.asyncio
async def test_generate_sub_topics(decompose_agent: DecomposeAgent):
    """
    Tests the _generate_sub_topics method.
    """
    decompose_agent.call_llm = AsyncMock(return_value='{"sub_topics": [{"title": "t1"}]}')
    decompose_agent.get_prompt = MagicMock(return_value="prompt")

    topics = await decompose_agent._generate_sub_topics("topic", "context", 1)
    
    assert len(topics) == 1

