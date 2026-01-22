#!/usr/bin/env python

"""
Tests for the BaseAgent.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.base_agent import BaseAgent
from src.services.llm.config import LLMConfig


class ConcreteAgent(BaseAgent):
    """A concrete implementation of BaseAgent for testing purposes."""

    async def process(self, *args, **kwargs):
        """Process method for the concrete agent."""
        pass  # pragma: no cover


@pytest.fixture
def base_agent() -> BaseAgent:
    """Provides a BaseAgent instance with mocked dependencies.

    Yields:
        BaseAgent: An instance of the BaseAgent.
    """
    with (
        patch("src.agents.base_agent.get_prompt_manager"),
        patch("src.agents.base_agent.get_logger"),
        patch(
            "src.agents.base_agent.get_agent_params",
            return_value={"temperature": 0.5, "max_tokens": 100},
        ),
    ):
        config = LLMConfig(model="test_model", binding="test_binding")
        agent = ConcreteAgent(
            module_name="test_module", agent_name="test_agent", config=config
        )
        yield agent


def test_base_agent_initialization(base_agent: BaseAgent):
    """Test that the BaseAgent is initialized correctly.

    Args:
        base_agent (BaseAgent): The BaseAgent instance.
    """
    assert base_agent.module_name == "test_module"
    assert base_agent.agent_name == "test_agent"
    assert base_agent.llm_config.model == "test_model"


def test_getters(base_agent: BaseAgent):
    """Test that the getter methods return the correct values.

    Args:
        base_agent (BaseAgent): The BaseAgent instance.
    """
    assert base_agent.get_model() == "test_model"
    assert base_agent.get_temperature() == 0.5
    assert base_agent.get_max_tokens() == 100


def test_token_tracking(base_agent: BaseAgent):
    """Test that the token tracking methods work correctly.

    Args:
        base_agent (BaseAgent): The BaseAgent instance.
    """
    base_agent.token_tracker = MagicMock()
    base_agent._track_tokens("test_model", "system", "user", "response")
    base_agent.token_tracker.add_usage.assert_called_once()

    base_agent.get_stats("test_module").reset()
    base_agent._track_usage("response", "test_stage")
    stats = base_agent.get_stats("test_module")
    assert stats.total_calls == 1


@pytest.mark.asyncio
async def test_call_llm(base_agent: BaseAgent):
    """Test that the call_llm method correctly calls the provider's complete method.

    Args:
        base_agent (BaseAgent): The BaseAgent instance.
    """
    base_agent.provider = MagicMock()
    base_agent.provider.complete = AsyncMock(
        return_value=MagicMock(content="test_response")
    )

    response = await base_agent.call_llm("user_prompt", "system_prompt")

    base_agent.provider.complete.assert_called_once()
    assert response == "test_response"


@pytest.mark.asyncio
async def test_stream_llm(base_agent: BaseAgent):
    """Test that the stream_llm method correctly calls the provider's stream method.

    Args:
        base_agent (BaseAgent): The BaseAgent instance.
    """

    async def mock_stream(*args, **kwargs):
        yield MagicMock(delta="test")
        yield MagicMock(delta=" response")

    base_agent.provider = MagicMock()
    base_agent.provider.stream = mock_stream

    chunks = [
        chunk async for chunk in base_agent.stream_llm("user_prompt", "system_prompt")
    ]

    assert "".join(chunks) == "test response"


def test_get_prompt(base_agent: BaseAgent):
    """Test that the get_prompt method correctly retrieves prompts.

    Args:
        base_agent (BaseAgent): The BaseAgent instance.
    """
    base_agent.prompts = {
        "system": "system_prompt",
        "section": {"field": "nested_prompt"},
    }

    assert base_agent.get_prompt("system") == "system_prompt"
    assert base_agent.get_prompt("section", "field") == "nested_prompt"
    assert base_agent.get_prompt("non_existent", fallback="default") == "default"
