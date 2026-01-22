from unittest.mock import AsyncMock, patch

import pytest

from src.agents.ideagen.idea_generation_workflow import IdeaGenerationWorkflow


@pytest.fixture
def idea_generation_workflow(tmp_path) -> IdeaGenerationWorkflow:
    """Provides an IdeaGenerationWorkflow instance with mocked dependencies.

    Args:
        tmp_path: Pytest fixture for a temporary directory.

    Yields:
        IdeaGenerationWorkflow: An instance of the IdeaGenerationWorkflow.
    """
    with (
        patch("src.agents.base_agent.get_prompt_manager"),
        patch("src.agents.base_agent.get_logger"),
    ):
        workflow = IdeaGenerationWorkflow(output_dir=tmp_path)
        yield workflow


@pytest.mark.asyncio
async def test_loose_filter(idea_generation_workflow: IdeaGenerationWorkflow):
    """Test the loose_filter method.

    Args:
        idea_generation_workflow (IdeaGenerationWorkflow): The IdeaGenerationWorkflow instance.
    """
    idea_generation_workflow.call_llm = AsyncMock(
        return_value='{"filtered_points": [{"knowledge_point": "kp1"}]}'
    )
    idea_generation_workflow._prompts = {
        "loose_filter_system": "",
        "loose_filter_user_template": "",
    }

    kps = [{"knowledge_point": "kp1"}, {"knowledge_point": "kp2"}]
    filtered = await idea_generation_workflow.loose_filter(kps)

    assert len(filtered) == 1
    assert filtered[0]["knowledge_point"] == "kp1"


@pytest.mark.asyncio
async def test_explore_ideas(idea_generation_workflow: IdeaGenerationWorkflow):
    """Test the explore_ideas method.

    Args:
        idea_generation_workflow (IdeaGenerationWorkflow): The IdeaGenerationWorkflow instance.
    """
    idea_generation_workflow.call_llm = AsyncMock(
        return_value='{"research_ideas": ["idea1", "idea2"]}'
    )
    idea_generation_workflow._prompts = {
        "explore_ideas_system": "",
        "explore_ideas_user_template": "",
    }

    ideas = await idea_generation_workflow.explore_ideas(
        {"knowledge_point": "kp1", "description": "desc"}
    )

    assert len(ideas) == 2


@pytest.mark.asyncio
async def test_strict_filter(idea_generation_workflow: IdeaGenerationWorkflow):
    """Test the strict_filter method.

    Args:
        idea_generation_workflow (IdeaGenerationWorkflow): The IdeaGenerationWorkflow instance.
    """
    idea_generation_workflow.call_llm = AsyncMock(return_value='{"kept_ideas": ["idea1"]}')
    idea_generation_workflow._prompts = {
        "strict_filter_system": "",
        "strict_filter_user_template": "",
    }

    kept = await idea_generation_workflow.strict_filter(
        {"knowledge_point": "kp1"}, ["idea1", "idea2"]
    )

    assert len(kept) == 1
    assert kept[0] == "idea1"


@pytest.mark.asyncio
async def test_generate_statement(
    idea_generation_workflow: IdeaGenerationWorkflow,
):
    """Test the generate_statement method.

    Args:
        idea_generation_workflow (IdeaGenerationWorkflow): The IdeaGenerationWorkflow instance.
    """
    idea_generation_workflow.call_llm = AsyncMock(return_value="statement")
    idea_generation_workflow._prompts = {
        "generate_statement_system": "",
        "generate_statement_user_template": "",
    }

    statement = await idea_generation_workflow.generate_statement(
        {"knowledge_point": "kp1"}, ["idea1"]
    )

    assert statement == "statement"


@pytest.mark.asyncio
async def test_process_workflow(
    idea_generation_workflow: IdeaGenerationWorkflow,
):
    """Test the overall workflow of the process method.

    Args:
        idea_generation_workflow (IdeaGenerationWorkflow): The IdeaGenerationWorkflow instance.
    """
    idea_generation_workflow.loose_filter = AsyncMock(
        return_value=[{"knowledge_point": "kp1", "description": ""}]
    )
    idea_generation_workflow.explore_ideas = AsyncMock(return_value=["idea1"])
    idea_generation_workflow.strict_filter = AsyncMock(return_value=["idea1"])
    idea_generation_workflow.generate_statement = AsyncMock(return_value="statement")

    markdown = await idea_generation_workflow.process(
        [{"knowledge_point": "kp1", "description": ""}]
    )

    assert "Research Ideas Generation Result" in markdown
    assert "statement" in markdown
