#!/usr/bin/env python

"""
Tests for the ToolAgent.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.solve.memory import (
    CitationMemory,
    SolveChainStep,
    SolveMemory,
    ToolCallRecord,
)
from src.agents.solve.solve_loop.tool_agent import ToolAgent


@pytest.fixture
def tool_agent():
    """
    Provides a ToolAgent instance with mocked dependencies.
    """
    with (
        patch("src.agents.base_agent.get_prompt_manager"),
        patch("src.agents.base_agent.get_logger"),
    ):
        config = {"system": {"language": "en"}}
        agent = ToolAgent(
            config=config, api_key="test_key", base_url="http://localhost:1234"
        )
        yield agent


@pytest.mark.asyncio
async def test_tool_agent_process_workflow(tool_agent: ToolAgent):
    """
    Tests the overall workflow of the ToolAgent's process method.
    """
    tool_agent._execute_single_call = AsyncMock(
        return_value=("raw_answer", {})
    )
    tool_agent._summarize_tool_result = AsyncMock(return_value="summary")

    step = SolveChainStep(step_id="step1", step_target="target1")
    step.tool_calls.append(
        ToolCallRecord(
            tool_type="rag_hybrid", query="test_query", cite_id="cite1"
        )
    )

    solve_memory = SolveMemory()
    solve_memory.create_chains([step])
    citation_memory = CitationMemory()
    citation_memory.add_citation("rag_hybrid", "test_query", cite_id="cite1")

    result = await tool_agent.process(
        step, solve_memory, citation_memory, "kb_name", "/tmp"
    )

    tool_agent._execute_single_call.assert_called_once()
    tool_agent._summarize_tool_result.assert_called_once()
    assert len(result["executed"]) == 1
    assert result["executed"][0]["status"] == "success"
    assert solve_memory.get_step("step1").tool_calls[0].status == "success"


@pytest.mark.asyncio
@patch("src.tools.rag_tool.rag_search", new_callable=AsyncMock)
@patch("src.tools.web_search.web_search")
@patch("src.tools.code_executor.run_code", new_callable=AsyncMock)
async def test_execute_single_call(
    mock_run_code, mock_web_search, mock_rag_search, tool_agent: ToolAgent
):
    """
    Tests that the _execute_single_call method correctly calls the different tools.
    """
    mock_rag_search.return_value = {"answer": "rag"}
    mock_web_search.return_value = {"answer": "web"}
    mock_run_code.return_value = {"stdout": "code"}

    record = ToolCallRecord(tool_type="rag_hybrid", query="q")
    answer, _ = await tool_agent._execute_single_call(
        record, "kb", "/tmp", "/tmp/artifacts", True
    )
    assert answer == "rag"

    record = ToolCallRecord(tool_type="web_search", query="q")
    answer, _ = await tool_agent._execute_single_call(
        record, "kb", "/tmp", "/tmp/artifacts", True
    )
    assert answer == "web"

    record = ToolCallRecord(tool_type="code_execution", query="q")
    tool_agent._generate_code_from_intent = AsyncMock(return_value="code")
    answer, _ = await tool_agent._execute_single_call(
        record, "kb", "/tmp", "/tmp/artifacts", True
    )
    assert "code" in answer


@pytest.mark.asyncio
async def test_summarize_tool_result(tool_agent: ToolAgent):
    """
    Tests that the _summarize_tool_result method correctly summarizes the tool result.
    """
    tool_agent.call_llm = AsyncMock(return_value="summary")
    tool_agent.get_prompt = MagicMock(return_value="prompt")

    summary = await tool_agent._summarize_tool_result(
        "tool", "query", "raw_answer"
    )

    assert summary == "summary"
