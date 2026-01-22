from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.co_writer.edit_agent import EditAgent


@pytest.fixture
def edit_agent() -> EditAgent:
    """Provides an EditAgent instance with mocked dependencies.

    Yields:
        EditAgent: An instance of the EditAgent.
    """
    with (
        patch("src.agents.base_agent.get_prompt_manager"),
        patch("src.agents.base_agent.get_logger"),
        patch("src.agents.co_writer.edit_agent.load_history", return_value=[]),
        patch("src.agents.co_writer.edit_agent.save_history"),
    ):
        agent = EditAgent()
        yield agent


@pytest.mark.asyncio
async def test_edit_agent_process_rewrite(edit_agent: EditAgent):
    """Test that the process method correctly calls the LLM for the rewrite action.

    Args:
        edit_agent (EditAgent): The EditAgent instance.
    """
    edit_agent.call_llm = AsyncMock(return_value="edited_text")

    result = await edit_agent.process("text", "instruction", "rewrite")

    edit_agent.call_llm.assert_called_once()
    assert result["edited_text"] == "edited_text"


@pytest.mark.asyncio
@patch("src.agents.co_writer.edit_agent.rag_search", new_callable=AsyncMock)
@patch("src.agents.co_writer.edit_agent.web_search")
async def test_edit_agent_process_with_source(
    mock_web_search, mock_rag_search, edit_agent: EditAgent
):
    """Test that the process method correctly uses RAG and web search.

    Args:
        mock_web_search: Mock object for the web_search function.
        mock_rag_search: Mock object for the rag_search function.
        edit_agent (EditAgent): The EditAgent instance.
    """
    edit_agent.call_llm = AsyncMock(return_value="edited_text")
    mock_rag_search.return_value = {"answer": "rag_context"}
    mock_web_search.return_value = {"answer": "web_context"}

    # Test RAG
    await edit_agent.process("text", "instruction", source="rag", kb_name="test_kb")
    mock_rag_search.assert_called_once()

    # Test Web Search
    await edit_agent.process("text", "instruction", source="web")
    mock_web_search.assert_called_once()


@pytest.mark.asyncio
async def test_auto_mark(edit_agent: EditAgent):
    """Test that the auto_mark method correctly calls the LLM.

    Args:
        edit_agent (EditAgent): The EditAgent instance.
    """
    edit_agent.call_llm = AsyncMock(return_value="marked_text")
    edit_agent.get_prompt = MagicMock(return_value="prompt")

    result = await edit_agent.auto_mark("text")

    edit_agent.call_llm.assert_called_once()
    assert result["marked_text"] == "marked_text"


@patch("src.agents.co_writer.edit_agent.save_history")
@pytest.mark.asyncio
async def test_history_saving(mock_save_history, edit_agent: EditAgent):
    """Test that the history is correctly saved after processing.

    Args:
        mock_save_history: Mock object for the save_history function.
        edit_agent (EditAgent): The EditAgent instance.
    """
    edit_agent.call_llm = AsyncMock(return_value="edited_text")

    await edit_agent.process("text", "instruction")

    mock_save_history.assert_called_once()
    history = mock_save_history.call_args[0][0]
    assert len(history) == 1
    assert history[0]["action"] == "rewrite"
