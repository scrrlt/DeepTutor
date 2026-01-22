from unittest.mock import AsyncMock, patch

import pytest

from src.agents.chat.chat_agent import ChatAgent


@pytest.fixture
def chat_agent() -> ChatAgent:
    """Provides a ChatAgent instance with mocked dependencies.

    Yields:
        ChatAgent: An instance of the ChatAgent.
    """
    with (
        patch("src.agents.base_agent.get_prompt_manager"),
        patch("src.agents.base_agent.get_logger"),
    ):
        agent = ChatAgent()
        yield agent


@pytest.mark.asyncio
async def test_chat_agent_process_no_stream(chat_agent: ChatAgent):
    """Test that the process method correctly calls call_llm when stream is False.

    Args:
        chat_agent (ChatAgent): The ChatAgent instance.
    """
    chat_agent.call_llm = AsyncMock(return_value="test_response")

    result = await chat_agent.process("test_message")

    chat_agent.call_llm.assert_called_once()
    assert result["response"] == "test_response"


@pytest.mark.asyncio
async def test_chat_agent_process_with_stream(chat_agent: ChatAgent):
    """Test that the process method correctly calls stream_llm when stream is True.

    Args:
        chat_agent (ChatAgent): The ChatAgent instance.
    """

    async def mock_stream_llm(*args, **kwargs):
        yield "test"
        yield " response"

    chat_agent.stream_llm = mock_stream_llm

    generator = await chat_agent.process("test_message", stream=True)

    chunks = [chunk async for chunk in generator]

    assert len(chunks) == 3  # 2 content chunks, 1 meta chunk
    assert chunks[0]["content"] == "test"
    assert chunks[1]["content"] == " response"
    assert chunks[2]["type"] == "meta"


@pytest.mark.asyncio
async def test_retrieve_context(chat_agent: ChatAgent):
    """Test that _retrieve_context correctly calls RAG and web search.

    Args:
        chat_agent (ChatAgent): The ChatAgent instance.
    """
    chat_agent._rag_search = AsyncMock(return_value={"answer": "rag_answer"})
    chat_agent._web_search = AsyncMock(
        return_value={"answer": "web_answer", "citations": []}
    )

    context, sources = await chat_agent._retrieve_context(
        "test_message", "test_kb", True, True
    )

    chat_agent._rag_search.assert_called_once_with("test_message", kb_name="test_kb")
    chat_agent._web_search.assert_called_once_with("test_message")
    assert "rag_answer" in context
    assert "web_answer" in context
    assert len(sources["rag"]) > 0
    assert len(sources["web"]) == 0


@pytest.mark.asyncio
async def test_retrieve_context_exceptions(chat_agent: ChatAgent):
    """Test that _retrieve_context correctly handles exceptions.

    Args:
        chat_agent (ChatAgent): The ChatAgent instance.
    """
    chat_agent._rag_search = AsyncMock(side_effect=Exception("RAG Error"))
    chat_agent._web_search = AsyncMock(side_effect=Exception("Web Search Error"))

    context, sources = await chat_agent._retrieve_context(
        "test_message", "test_kb", True, True
    )

    assert context == ""
    assert len(sources["rag"]) == 0
    assert len(sources["web"]) == 0


def test_truncate_history(chat_agent: ChatAgent):
    """Test that _truncate_history correctly truncates the history.

    Args:
        chat_agent (ChatAgent): The ChatAgent instance.
    """
    history = [
        {
            "role": "user",
            "content": "This is a long message that should be truncated.",
        },
        {"role": "assistant", "content": "This is another long message."},
        {"role": "user", "content": "This is a short message."},
    ]
    chat_agent.max_history_tokens = 20

    truncated_history = chat_agent._truncate_history(history)

    assert len(truncated_history) == 1
    assert truncated_history[0]["content"] == "This is a short message."
