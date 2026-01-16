"""
Tests for the research's RephraseAgent.
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from src.agents.research.agents.rephrase_agent import RephraseAgent


@pytest.fixture
def rephrase_agent():
    """
    Provides a RephraseAgent instance with mocked dependencies.
    """
    with patch('src.agents.base_agent.get_prompt_manager'), \
         patch('src.agents.base_agent.get_logger'):
        agent = RephraseAgent(config={})
        yield agent


@pytest.mark.asyncio
async def test_rephrase_agent_process_workflow(rephrase_agent: RephraseAgent):
    """
    Tests that the process method correctly calls the LLM and returns the rephrased topic.
    """
    rephrase_agent.call_llm = AsyncMock(return_value='{"topic": "rephrased"}')
    rephrase_agent.get_prompt = MagicMock(return_value="prompt")

    result = await rephrase_agent.process("original")
    
    rephrase_agent.call_llm.assert_called_once()
    assert result["topic"] == "rephrased"


@pytest.mark.asyncio
async def test_check_user_satisfaction(rephrase_agent: RephraseAgent):
    """
    Tests that the check_user_satisfaction method correctly calls the LLM.
    """
    rephrase_agent.call_llm = AsyncMock(return_value='{"user_satisfied": true, "should_continue": false}')
    rephrase_agent.get_prompt = MagicMock(return_value="prompt")

    result = await rephrase_agent.check_user_satisfaction({}, "I am happy")
    
    rephrase_agent.call_llm.assert_called_once()
    assert result["user_satisfied"] is True

def test_conversation_history(rephrase_agent: RephraseAgent):
    """
    Tests that the conversation history is correctly managed.
    """
    rephrase_agent.reset_history()
    assert len(rephrase_agent.conversation_history) == 0
    
    rephrase_agent.conversation_history.append({"role": "user", "content": "hi"})
    formatted = rephrase_agent._format_conversation_history()
    
    assert "[User - Initial Input]" in formatted
    assert "hi" in formatted

