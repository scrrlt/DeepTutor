"""
Tests for the research's NoteAgent.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.research.agents.note_agent import NoteAgent


@pytest.fixture
def note_agent():
    """
    Provides a NoteAgent instance with mocked dependencies.
    """
    with (
        patch("src.agents.base_agent.get_prompt_manager"),
        patch("src.agents.base_agent.get_logger"),
    ):
        agent = NoteAgent(config={})
        yield agent


@pytest.mark.asyncio
async def test_note_agent_process_workflow(note_agent: NoteAgent):
    """
    Tests that the process method correctly calls the _generate_summary method.
    """
    note_agent._generate_summary = AsyncMock(return_value="summary")

    trace = await note_agent.process("tool", "query", "raw", "cite1")

    note_agent._generate_summary.assert_called_once()
    assert trace.summary == "summary"
    assert trace.cite_id == "cite1"


@pytest.mark.asyncio
async def test_generate_summary(note_agent: NoteAgent):
    """
    Tests that the _generate_summary method correctly calls the LLM.
    """
    note_agent.call_llm = AsyncMock(return_value='{"summary": "summary"}')
    note_agent.get_prompt = MagicMock(return_value="prompt")

    summary = await note_agent._generate_summary("tool", "query", "raw")

    note_agent.call_llm.assert_called_once()
    assert summary == "summary"
