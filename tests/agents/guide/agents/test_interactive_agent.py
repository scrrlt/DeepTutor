#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests for the guide's InteractiveAgent.
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from src.agents.guide.agents.interactive_agent import InteractiveAgent


@pytest.fixture
def interactive_agent():
    """
    Provides an InteractiveAgent instance with mocked dependencies.
    """
    with patch('src.agents.base_agent.get_prompt_manager'), \
         patch('src.agents.base_agent.get_logger'):
        agent = InteractiveAgent(api_key="test_key", base_url="http://localhost:1234")
        yield agent


@pytest.mark.asyncio
async def test_interactive_agent_process_workflow(interactive_agent: InteractiveAgent):
    """
    Tests that the process method correctly calls the LLM and returns the HTML.
    """
    interactive_agent.call_llm = AsyncMock(return_value="<html></html>")
    interactive_agent.get_prompt = MagicMock(return_value="prompt")

    knowledge = {"knowledge_title": "t", "knowledge_summary": "s", "user_difficulty": "d"}
    
    result = await interactive_agent.process(knowledge)

    interactive_agent.call_llm.assert_called_once()
    assert result["success"] is True
    assert "<html>" in result["html"]


@pytest.mark.asyncio
async def test_interactive_agent_process_bug_fix(interactive_agent: InteractiveAgent):
    """
    Tests that the process method correctly handles a bug fix request.
    """
    interactive_agent.call_llm = AsyncMock(return_value="<html>fixed</html>")
    interactive_agent.get_prompt = MagicMock(return_value="prompt")
    
    knowledge = {"knowledge_title": "t", "knowledge_summary": "s", "user_difficulty": "d"}
    
    result = await interactive_agent.process(knowledge, retry_with_bug="bug")

    interactive_agent.call_llm.assert_called_once()
    assert "fixed" in result["html"]


def test_extract_html(interactive_agent: InteractiveAgent):
    """
    Tests that the _extract_html method correctly extracts the HTML.
    """
    response = "```html\n<html>\n</html>\n```"
    html = interactive_agent._extract_html(response)
    assert "<html>" in html

    response = "<html></html>"
    html = interactive_agent._extract_html(response)
    assert "<html>" in html


def test_validate_html(interactive_agent: InteractiveAgent):
    """
    Tests that the _validate_html method correctly validates the HTML.
    """
    assert interactive_agent._validate_html("<html></html>") is True
    assert interactive_agent._validate_html("no html") is False


def test_generate_fallback_html(interactive_agent: InteractiveAgent):
    """
    Tests that the _generate_fallback_html method correctly generates the fallback HTML.
    """
    knowledge = {"knowledge_title": "t", "knowledge_summary": "s", "user_difficulty": "d"}
    html = interactive_agent._generate_fallback_html(knowledge)
    assert "t" in html
    assert "s" in html
    assert "d" in html
