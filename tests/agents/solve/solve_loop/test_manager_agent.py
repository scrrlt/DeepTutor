#!/usr/bin/env python

"""
Tests for the ManagerAgent.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.solve.memory import (
    InvestigateMemory,
    SolveMemory,
)
from src.agents.solve.solve_loop.manager_agent import ManagerAgent


@pytest.fixture
def manager_agent():
    """
    Provides a ManagerAgent instance with mocked dependencies.
    """
    with (
        patch("src.agents.base_agent.get_prompt_manager"),
        patch("src.agents.base_agent.get_logger"),
    ):
        config = {"system": {"language": "en"}}
        agent = ManagerAgent(config=config, api_key="test_key", base_url="http://localhost:1234")
        yield agent


@pytest.mark.asyncio
async def test_manager_agent_process_workflow(manager_agent: ManagerAgent):
    """
    Tests the overall workflow of the ManagerAgent's process method.
    """
    manager_agent.call_llm = AsyncMock(
        return_value='{"steps": [{"step_id": "S1", "target": "t1"}]}'
    )

    investigate_memory = InvestigateMemory(user_question="test question")
    solve_memory = SolveMemory(user_question="test question")

    result = await manager_agent.process("test question", investigate_memory, solve_memory)

    manager_agent.call_llm.assert_called_once()
    assert result["steps_count"] == 1
    assert len(solve_memory.solve_chains) == 1
    assert solve_memory.solve_chains[0].step_id == "S1"


@pytest.mark.asyncio
async def test_manager_agent_process_invalid_json(manager_agent: ManagerAgent):
    """
    Tests that the process method correctly handles invalid JSON.
    """
    manager_agent.call_llm = AsyncMock(return_value="invalid json")

    investigate_memory = InvestigateMemory(user_question="test question")
    solve_memory = SolveMemory(user_question="test question")

    with pytest.raises(ValueError):
        await manager_agent.process("test question", investigate_memory, solve_memory)


@pytest.mark.asyncio
async def test_manager_agent_process_existing_steps(
    manager_agent: ManagerAgent,
):
    """
    Tests that the process method correctly handles existing steps.
    """
    investigate_memory = InvestigateMemory(user_question="test question")
    solve_memory = SolveMemory(user_question="test question")
    solve_memory.create_chains([MagicMock()])

    result = await manager_agent.process("test question", investigate_memory, solve_memory)

    assert result["has_steps"] is True
    assert result["steps_count"] == 1
    assert "skipping" in result["message"]
