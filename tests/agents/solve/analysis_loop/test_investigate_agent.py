#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests for the InvestigateAgent.
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from src.agents.solve.analysis_loop.investigate_agent import InvestigateAgent
from src.agents.solve.memory import InvestigateMemory, CitationMemory


@pytest.fixture
def investigate_agent():
    """
    Provides an InvestigateAgent instance with mocked dependencies.
    """
    with patch('src.agents.base_agent.get_prompt_manager'), \
         patch('src.agents.base_agent.get_logger'):
        config = {
            "system": {"language": "en"},
            "tools": {"web_search": {"enabled": True}},
            "solve": {
                "agents": {
                    "investigate_agent": {
                        "max_actions_per_round": 1,
                        "max_iterations": 3,
                    }
                }
            }
        }
        agent = InvestigateAgent(config=config, api_key="test_key", base_url="http://localhost:1234")
        yield agent


@pytest.mark.asyncio
async def test_investigate_agent_process_workflow(investigate_agent: InvestigateAgent):
    """
    Tests the overall workflow of the InvestigateAgent's process method.
    """
    investigate_agent.call_llm = AsyncMock(return_value='{"reasoning": "test reasoning", "plan": [{"tool": "rag_hybrid", "query": "test query"}]}')
    investigate_agent._execute_single_action = AsyncMock(return_value=MagicMock(cite_id="cite1"))
    
    memory = InvestigateMemory(user_question="test question")
    citation_memory = CitationMemory(output_dir="/tmp/test")

    result = await investigate_agent.process("test question", memory, citation_memory)

    investigate_agent.call_llm.assert_called_once()
    investigate_agent._execute_single_action.assert_called_once()
    assert result["reasoning"] == "test reasoning"
    assert len(result["knowledge_item_ids"]) == 1
    assert result["knowledge_item_ids"][0] == "cite1"
    assert len(result["actions"]) == 1
    assert result["actions"][0]["tool_type"] == "rag_hybrid"


@pytest.mark.asyncio
async def test_investigate_agent_process_empty_plan(investigate_agent: InvestigateAgent):
    """
    Tests that the process method correctly handles an empty plan.
    """
    investigate_agent.call_llm = AsyncMock(return_value='{"reasoning": "test reasoning", "plan": []}')
    
    memory = InvestigateMemory(user_question="test question")
    citation_memory = CitationMemory(output_dir="/tmp/test")

    result = await investigate_agent.process("test question", memory, citation_memory)

    investigate_agent.call_llm.assert_called_once()
    assert result["should_stop"] is True

def test_build_system_prompt_web_search_disabled(investigate_agent: InvestigateAgent):
    """
    Tests that the _build_system_prompt method correctly removes the web search part when it is disabled.
    """
    investigate_agent.enable_web_search = False
    
    # Mock the get_prompt method to return a prompt with web_search
    investigate_agent.get_prompt = MagicMock(side_effect=lambda key: {
        "system": "This is a system prompt with `web_search`.",
        "web_search_description": "`web_search`",
        "web_search_disabled": "Web search is disabled."
    }[key])
    
    prompt = investigate_agent._build_system_prompt()

    assert "Web search is disabled." in prompt
    assert "`web_search`" not in prompt or "This is a system prompt with Web search is disabled.." in prompt
