"""
Tests for the guide's SummaryAgent.
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from src.agents.guide.agents.summary_agent import SummaryAgent


@pytest.fixture
def summary_agent():
    """
    Provides a SummaryAgent instance with mocked dependencies.
    """
    with patch('src.agents.base_agent.get_prompt_manager'), \
         patch('src.agents.base_agent.get_logger'):
        agent = SummaryAgent(api_key="test_key", base_url="http://localhost:1234")
        yield agent


@pytest.mark.asyncio
async def test_summary_agent_process_workflow(summary_agent: SummaryAgent):
    """
    Tests that the process method correctly calls the LLM and returns the summary.
    """
    summary_agent.call_llm = AsyncMock(return_value="summary")
    summary_agent.get_prompt = MagicMock(return_value="prompt")

    points = [{"knowledge_title": "t", "knowledge_summary": "s", "user_difficulty": "d"}]
    history = [{"role": "user", "content": "q"}]
    
    result = await summary_agent.process("nname", points, history)

    summary_agent.call_llm.assert_called_once()
    assert result["success"] is True
    assert result["summary"] == "summary"


def test_format_knowledge_points(summary_agent: SummaryAgent):
    """
    Tests that the _format_knowledge_points method correctly formats the knowledge points.
    """
    points = [{"knowledge_title": "t1", "knowledge_summary": "s1", "user_difficulty": "d1"}]
    formatted = summary_agent._format_knowledge_points(points)
    
    assert "Knowledge Point 1: t1" in formatted
    assert "Content Summary: s1" in formatted


def test_format_chat_history(summary_agent: SummaryAgent):
    """
    Tests that the _format_chat_history method correctly formats the chat history.
    """
    history = [
        {"role": "user", "content": "q1", "knowledge_index": 0},
        {"role": "assistant", "content": "a1", "knowledge_index": 0},
    ]
    formatted = summary_agent._format_chat_history(history)
    
    assert "**User Question**: q1" in formatted
    assert "**Assistant Answer**: a1" in formatted

