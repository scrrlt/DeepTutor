from unittest.mock import MagicMock, patch

import pytest
from src.agents.base_agent import BaseAgent
from src.config.config import LLMConfig


class TestAgent(BaseAgent):
    """A concrete implementation of BaseAgent for testing internal methods."""

    async def process(self):
        """Process method for the test agent."""
        pass

@pytest.fixture
def base_agent() -> BaseAgent:
    """Provides a BaseAgent instance with mocked dependencies for internal testing.

    Yields:
        BaseAgent: An instance of the BaseAgent.
    """
    config = LLMConfig(model="test-model", binding="openai")
    with patch("src.agents.base_agent.LLMFactory.get_provider"), patch(
        "src.agents.base_agent.get_prompt_manager"
    ) as mock_pm:
        mock_pm.return_value.load_prompts.return_value = {
            "system": "Default System",
            "nested": {"key": "Nested Value"},
        }
        yield TestAgent(module_name="test", agent_name="tester", config=config)

def test_prompt_lookup_simple(base_agent: BaseAgent):
    """Test simple key lookup for prompts.

    Args:
        base_agent (BaseAgent): The BaseAgent instance.
    """
    assert base_agent.get_prompt("system") == "Default System"


def test_prompt_lookup_nested(base_agent: BaseAgent):
    """Test 'section.key' dot notation lookup for prompts.

    Args:
        base_agent (BaseAgent): The BaseAgent instance.
    """
    assert base_agent.get_prompt("nested.key") == "Nested Value"


def test_prompt_fallback(base_agent: BaseAgent):
    """Test fallback mechanism when a prompt key is missing.

    Args:
        base_agent (BaseAgent): The BaseAgent instance.
    """
    assert base_agent.get_prompt("missing", fallback="Fallback") == "Fallback"
    assert (
        base_agent.get_prompt("nested.missing", fallback="Deep Fallback") == "Deep Fallback"
    )


def test_track_usage_delegation(base_agent: BaseAgent):
    """Verify that stats are correctly sent to the LLMStats tracker.

    Args:
        base_agent (BaseAgent): The BaseAgent instance.
    """
    mock_stats = MagicMock()
    with patch.object(BaseAgent, "get_stats", return_value=mock_stats):
        base_agent._track_usage("response text", stage="testing")

        mock_stats.add_call.assert_called_once()
        call_args = mock_stats.add_call.call_args.kwargs
        assert call_args["model"] == "test-model"
        assert call_args["response"] == "response text"