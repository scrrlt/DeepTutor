from unittest.mock import AsyncMock, patch

import pytest

from src.agents.solve.analysis_loop.note_agent import NoteAgent
from src.agents.solve.memory import (
    CitationMemory,
    InvestigateMemory,
    KnowledgeItem,
)


@pytest.fixture
def note_agent() -> NoteAgent:
    """Provides a NoteAgent instance with mocked dependencies.

    Yields:
        NoteAgent: An instance of the NoteAgent.
    """
    with (
        patch("src.agents.base_agent.get_prompt_manager"),
        patch("src.agents.base_agent.get_logger"),
    ):
        config = {
            "system": {"language": "en"},
        }
        agent = NoteAgent(
            config=config, api_key="test_key", base_url="http://localhost:1234"
        )
        yield agent


@pytest.mark.asyncio
async def test_note_agent_process_workflow(note_agent: NoteAgent):
    """Test the overall workflow of the NoteAgent's process method.

    Args:
        note_agent (NoteAgent): The NoteAgent instance.
    """
    note_agent.call_llm = AsyncMock(
        return_value='{"summary": "test summary", "citations": []}'
    )

    memory = InvestigateMemory(user_question="test question")
    knowledge_item = KnowledgeItem(
        cite_id="cite1",
        tool_type="test_tool",
        query="test query",
        raw_result="test result",
    )
    memory.add_knowledge(knowledge_item)

    citation_memory = CitationMemory(output_dir="/tmp/test")

    result = await note_agent.process(
        "test question", memory, ["cite1"], citation_memory
    )

    note_agent.call_llm.assert_called_once()
    assert result["success"] is True
    assert result["processed_items"] == 1
    assert memory.knowledge_chain[0].summary == "test summary"


@pytest.mark.asyncio
async def test_note_agent_process_invalid_json(note_agent: NoteAgent):
    """Test that the process method correctly handles invalid JSON.

    Args:
        note_agent (NoteAgent): The NoteAgent instance.
    """
    note_agent.call_llm = AsyncMock(return_value="invalid json")

    memory = InvestigateMemory(user_question="test question")
    knowledge_item = KnowledgeItem(
        cite_id="cite1",
        tool_type="test_tool",
        query="test query",
        raw_result="test result",
    )
    memory.add_knowledge(knowledge_item)

    citation_memory = CitationMemory(output_dir="/tmp/test")

    result = await note_agent.process(
        "test question", memory, ["cite1"], citation_memory
    )

    note_agent.call_llm.assert_called_once()
    assert result["success"] is False
    assert len(result["failed"]) == 1
    assert result["failed"][0]["cite_id"] == "cite1"


@pytest.mark.asyncio
async def test_note_agent_process_knowledge_not_found(note_agent: NoteAgent):
    """Test that the process method handles missing knowledge items.

    Args:
        note_agent (NoteAgent): The NoteAgent instance.
    """
    memory = InvestigateMemory(user_question="test question")
    citation_memory = CitationMemory(output_dir="/tmp/test")

    result = await note_agent.process(
        "test question", memory, ["cite1"], citation_memory
    )

    assert result["success"] is False
    assert len(result["failed"]) == 1
    assert result["failed"][0]["cite_id"] == "cite1"
    assert "not found" in result["failed"][0]["reason"]
