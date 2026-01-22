#!/usr/bin/env python

"""
Tests for the guide's ChatAgent.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.guide.agents.chat_agent import ChatAgent


@pytest.fixture
def chat_agent():
    """
    Provides a ChatAgent instance with mocked dependencies.
    """
    with (
        patch("src.agents.base_agent.get_prompt_manager"),
        patch("src.agents.base_agent.get_logger"),
    ):
        agent = ChatAgent(api_key="test_key", base_url="http://localhost:1234")
        yield agent


@pytest.mark.asyncio
async def test_chat_agent_process_workflow(chat_agent: ChatAgent):
    """
    Tests that the process method correctly calls the LLM and returns the answer.
    """
    chat_agent.call_llm = AsyncMock(return_value="assistant_response")
    chat_agent.get_prompt = MagicMock(return_value="prompt")

    knowledge = {
        "knowledge_title": "t",
        "knowledge_summary": "s",
        "user_difficulty": "d",
    }
    history = [{"role": "user", "content": "q"}]

    result = await chat_agent.process(knowledge, history, "user_question")

    chat_agent.call_llm.assert_called_once()
    assert result["success"] is True
    assert result["answer"] == "assistant_response"


@pytest.mark.asyncio
async def test_chat_agent_process_empty_question(chat_agent: ChatAgent):
    """
    Tests that the process method handles an empty question.
    """
    result = await chat_agent.process({}, [], "")
    assert result["success"] is False
    assert "empty" in result["error"]


def test_format_chat_history(chat_agent: ChatAgent):
    """
    Tests that the _format_chat_history method correctly formats the chat history.
    """
    history = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
    ]
    formatted = chat_agent._format_chat_history(history)

    assert "**User**: q1" in formatted
    assert "**Assistant**: a1" in formatted
