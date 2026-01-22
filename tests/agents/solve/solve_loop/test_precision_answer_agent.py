#!/usr/bin/env python

"""
Tests for the PrecisionAnswerAgent.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.solve.solve_loop.precision_answer_agent import (
    PrecisionAnswerAgent,
)


@pytest.fixture
def precision_answer_agent():
    """
    Provides a PrecisionAnswerAgent instance with mocked dependencies.
    """
    with (
        patch("src.agents.base_agent.get_prompt_manager"),
        patch("src.agents.base_agent.get_logger"),
    ):
        config = {"system": {"language": "en"}}
        agent = PrecisionAnswerAgent(
            config=config, api_key="test_key", base_url="http://localhost:1234"
        )
        yield agent


@pytest.mark.asyncio
async def test_precision_answer_agent_process_workflow(
    precision_answer_agent: PrecisionAnswerAgent,
):
    """
    Tests the overall workflow of the PrecisionAnswerAgent's process method.
    """
    precision_answer_agent._should_generate = AsyncMock(return_value={"needs_precision": True})
    precision_answer_agent._generate_precision_answer = AsyncMock(return_value="precision_answer")

    result = await precision_answer_agent.process("question", "detailed_answer")

    precision_answer_agent._should_generate.assert_called_once()
    precision_answer_agent._generate_precision_answer.assert_called_once()
    assert result["needs_precision"] is True
    assert result["precision_answer"] == "precision_answer"


@pytest.mark.asyncio
async def test_should_generate(precision_answer_agent: PrecisionAnswerAgent):
    """
    Tests the _should_generate method.
    """
    precision_answer_agent.get_prompt = MagicMock(return_value="prompt")
    precision_answer_agent.call_llm = AsyncMock(return_value="Y")

    result = await precision_answer_agent._should_generate("question", True)
    assert result["needs_precision"] is True

    precision_answer_agent.call_llm = AsyncMock(return_value="N")
    result = await precision_answer_agent._should_generate("question", True)
    assert result["needs_precision"] is False


@pytest.mark.asyncio
async def test_generate_precision_answer(
    precision_answer_agent: PrecisionAnswerAgent,
):
    """
    Tests the _generate_precision_answer method.
    """
    precision_answer_agent.get_prompt = MagicMock(return_value="prompt")
    precision_answer_agent.call_llm = AsyncMock(return_value="precision_answer")

    result = await precision_answer_agent._generate_precision_answer(
        "question", "detailed_answer", True
    )
    assert result == "precision_answer"
