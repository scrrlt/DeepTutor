# tests/agents/test_base_agent_internals.py
import pytest
from unittest.mock import MagicMock, patch
from src.agents.base_agent import BaseAgent
from src.config.config import LLMConfig

# Concrete implementation for testing abstract class
class TestAgent(BaseAgent):
    async def process(self):
        pass

@pytest.fixture
def base_agent():
    config = LLMConfig(model="test-model", binding="openai")
    with patch("src.agents.base_agent.LLMFactory.get_provider"):
        with patch("src.agents.base_agent.get_prompt_manager") as mock_pm:
            # Mock loaded prompts
            mock_pm.return_value.load_prompts.return_value = {
                "system": "Default System",
                "nested": {
                    "key": "Nested Value"
                }
            }
            return TestAgent(module_name="test", agent_name="tester", config=config)

def test_prompt_lookup_simple(base_agent):
    """Test simple key lookup."""
    assert base_agent.get_prompt("system") == "Default System"

def test_prompt_lookup_nested(base_agent):
    """Test 'section.key' dot notation lookup."""
    assert base_agent.get_prompt("nested.key") == "Nested Value"

def test_prompt_fallback(base_agent):
    """Test fallback when key is missing."""
    assert base_agent.get_prompt("missing", fallback="Fallback") == "Fallback"
    assert base_agent.get_prompt("nested.missing", fallback="Deep Fallback") == "Deep Fallback"

def test_track_usage_delegation(base_agent):
    """Verify stats are sent to the LLMStats tracker."""
    # Mock the shared stats class
    mock_stats = MagicMock()
    with patch.object(BaseAgent, "get_stats", return_value=mock_stats):
        base_agent._track_usage("response text", stage="testing")
        
        mock_stats.add_call.assert_called_once()
        call_args = mock_stats.add_call.call_args.kwargs
        assert call_args["model"] == "test-model"
        assert call_args["response"] == "response text"