#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests for the SolveAgent.
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from src.agents.solve.solve_loop.solve_agent import SolveAgent
from src.agents.solve.memory import SolveMemory, InvestigateMemory, CitationMemory, SolveChainStep


@pytest.fixture
def solve_agent():
    """
    Provides a SolveAgent instance with mocked dependencies.
    """
    with patch('src.agents.base_agent.get_prompt_manager'), \
         patch('src.agents.base_agent.get_logger'):
        config = {"system": {"language": "en"}}
        agent = SolveAgent(config=config, api_key="test_key", base_url="http://localhost:1234")
        yield agent


@pytest.mark.asyncio
async def test_solve_agent_process_workflow(solve_agent: SolveAgent):
    """
    Tests the overall workflow of the SolveAgent's process method.
    """
    solve_agent.call_llm = AsyncMock(return_value='{"tool_calls": [{"type": "rag_hybrid", "intent": "test query"}]}')
    
    step = SolveChainStep(step_id="step1", step_target="target1")
    solve_memory = SolveMemory()
    solve_memory.create_chains([step])
    investigate_memory = InvestigateMemory()
    citation_memory = CitationMemory()

    result = await solve_agent.process("question", step, solve_memory, investigate_memory, citation_memory)

    solve_agent.call_llm.assert_called_once()
    assert len(result["requested_calls"]) == 1
    assert result["requested_calls"][0]["tool_type"] == "rag_hybrid"
    assert len(solve_memory.get_step("step1").tool_calls) == 1


@pytest.mark.asyncio
async def test_solve_agent_process_invalid_json(solve_agent: SolveAgent):
    """
    Tests that the process method correctly handles invalid JSON.
    """
    solve_agent.call_llm = AsyncMock(return_value='invalid json')
    
    step = SolveChainStep(step_id="step1", step_target="target1")
    solve_memory = SolveMemory()
    solve_memory.create_chains([step])
    investigate_memory = InvestigateMemory()
    citation_memory = CitationMemory()

    with pytest.raises(ValueError):
        await solve_agent.process("question", step, solve_memory, investigate_memory, citation_memory)


def test_parse_tool_plan(solve_agent: SolveAgent):
    """
    Tests that the _parse_tool_plan method correctly parses the tool plan.
    """
    response = '{"tool_calls": [{"type": "rag_hybrid", "intent": "q1"}, {"type": "finish", "intent": ""}]}'
    plan = solve_agent._parse_tool_plan(response)
    
    assert len(plan) == 2
    assert plan[0]["type"] == "rag_hybrid"
    assert plan[1]["type"] == "finish"
