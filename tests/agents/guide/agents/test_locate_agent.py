#!/usr/bin/env python

"""
Tests for the guide's LocateAgent.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.guide.agents.locate_agent import LocateAgent


@pytest.fixture
def locate_agent():
    """
    Provides a LocateAgent instance with mocked dependencies.
    """
    with (
        patch("src.agents.base_agent.get_prompt_manager"),
        patch("src.agents.base_agent.get_logger"),
    ):
        agent = LocateAgent(api_key="test_key", base_url="http://localhost:1234")
        yield agent


@pytest.mark.asyncio
async def test_locate_agent_process_workflow(locate_agent: LocateAgent):
    """
    Tests that the process method correctly calls the LLM and returns the knowledge points.
    """
    locate_agent.call_llm = AsyncMock(
        return_value='{"knowledge_points": [{"knowledge_title": "kp1"}]}'
    )
    locate_agent.get_prompt = MagicMock(return_value="prompt")

    records = [{"type": "text", "title": "t", "user_query": "q", "output": "o"}]

    result = await locate_agent.process("nid", "nname", records)

    locate_agent.call_llm.assert_called_once()
    assert result["success"] is True
    assert len(result["knowledge_points"]) == 1
    assert result["knowledge_points"][0]["knowledge_title"] == "kp1"


@pytest.mark.asyncio
async def test_locate_agent_process_no_records(locate_agent: LocateAgent):
    """
    Tests that the process method handles no records.
    """
    result = await locate_agent.process("nid", "nname", [])
    assert result["success"] is False
    assert "No records" in result["error"]


def test_format_records(locate_agent: LocateAgent):
    """
    Tests that the _format_records method correctly formats the records.
    """
    records = [
        {"type": "text", "title": "t1", "user_query": "q1", "output": "o1"},
        {"type": "code", "title": "t2", "user_query": "q2", "output": "o2"},
    ]
    formatted = locate_agent._format_records(records)

    assert "Record 1 [TEXT]" in formatted
    assert "Record 2 [CODE]" in formatted
    assert "t1" in formatted
    assert "o2" in formatted
