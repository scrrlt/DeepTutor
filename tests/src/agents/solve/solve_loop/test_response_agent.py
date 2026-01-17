#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests for the ResponseAgent.
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from src.agents.solve.solve_loop.response_agent import ResponseAgent
from src.agents.solve.memory import SolveMemory, InvestigateMemory, CitationMemory, SolveChainStep


@pytest.fixture
def response_agent():
    """
    Provides a ResponseAgent instance with mocked dependencies.
    """
    with patch('src.agents.base_agent.get_prompt_manager'), \
         patch('src.agents.base_agent.get_logger'):
        config = {"system": {"language": "en", "enable_citations": True}}
        agent = ResponseAgent(config=config, api_key="test_key", base_url="http://localhost:1234")
        yield agent


@pytest.mark.asyncio
async def test_response_agent_process_workflow(response_agent: ResponseAgent):
    """
    Tests the overall workflow of the ResponseAgent's process method.
    """
    response_agent.call_llm = AsyncMock(return_value="test response [cite1]")
    
    step = SolveChainStep(step_id="step1", step_target="target1", available_cite=["[cite1]"])
    solve_memory = SolveMemory()
    solve_memory.create_chains([step])
    investigate_memory = InvestigateMemory()
    citation_memory = CitationMemory()

    result = await response_agent.process("question", step, solve_memory, investigate_memory, citation_memory)

    response_agent.call_llm.assert_called_once()
    assert result["step_response"] == "test response [cite1]"
    assert result["used_citations"] == ["[cite1]"]
    assert solve_memory.get_step("step1").status == "done"


def test_build_system_prompt_citations_disabled(response_agent: ResponseAgent):
    """
    Tests that the _build_system_prompt method correctly handles disabled citations.
    """
    response_agent.enable_citations = False
    response_agent.get_prompt = MagicMock(return_value="prompt")
    
    prompt = response_agent._build_system_prompt([])
    
    assert "citation" in prompt.lower()
    assert "disabled" in prompt.lower()


def test_extract_used_citations(response_agent: ResponseAgent):
    """
    Tests that the _extract_used_citations method correctly extracts the used citations.
    """
    content = "This is a response with [cite1] and [cite2]."
    step = SolveChainStep(step_id="s1", step_target="t1", available_cite=["[cite1]", "[cite2]", "[cite3]"])
    
    used = response_agent._extract_used_citations(content, step)
    
    assert used == ["[cite1]", "[cite2]"]
