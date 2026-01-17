from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from src.agents.question.coordinator import AgentCoordinator
from src.utils.config import load_config_with_main


@pytest.fixture
def coordinator(tmp_path) -> AgentCoordinator:
    """Provides an AgentCoordinator instance with mocked dependencies.

    Args:
        tmp_path: Pytest fixture for a temporary directory.

    Yields:
        AgentCoordinator: An instance of the AgentCoordinator.
    """
    with patch(
        "src.agents.question.coordinator.load_config_with_main",
        return_value={"question": {}, "paths": {}},
    ), patch("src.agents.question.coordinator.get_logger"):
        coord = AgentCoordinator(output_dir=str(tmp_path))
        yield coord


@pytest.mark.asyncio
async def test_generate_question(coordinator: AgentCoordinator):
    """Test the generate_question method.

    Args:
        coordinator (AgentCoordinator): The AgentCoordinator instance.
    """
    with patch.object(coordinator, "_create_retrieve_agent") as mock_create_retrieve, patch.object(
        coordinator, "_create_generate_agent"
    ) as mock_create_generate, patch.object(
        coordinator, "_create_relevance_analyzer"
    ) as mock_create_analyzer:
        mock_retrieve_agent = AsyncMock()
        mock_retrieve_agent.process.return_value = {"has_content": True, "summary": "context"}
        mock_create_retrieve.return_value = mock_retrieve_agent

        mock_generate_agent = AsyncMock()
        mock_generate_agent.process.return_value = {"success": True, "question": {}}
        mock_create_generate.return_value = mock_generate_agent

        mock_analyzer = AsyncMock()
        mock_analyzer.process.return_value = {"relevance": "high", "kb_coverage": "good"}
        mock_create_analyzer.return_value = mock_analyzer

        result = await coordinator.generate_question({"knowledge_point": "kp1"})

        assert result["success"] is True
        assert "question" in result
        assert "validation" in result


@pytest.mark.asyncio
async def test_generate_questions_custom(coordinator: AgentCoordinator):
    """Test the generate_questions_custom method.

    Args:
        coordinator (AgentCoordinator): The AgentCoordinator instance.
    """
    with patch.object(coordinator, "_create_retrieve_agent") as mock_create_retrieve, patch.object(
        coordinator, "_create_generate_agent"
    ) as mock_create_generate, patch.object(
        coordinator, "_create_relevance_analyzer"
    ) as mock_create_analyzer, patch.object(
        coordinator, "_generate_question_plan", new_callable=AsyncMock
    ) as mock_generate_plan:
        mock_retrieve_agent = AsyncMock()
        mock_retrieve_agent.process.return_value = {
            "has_content": True,
            "summary": "context",
            "queries": [],
        }
        mock_create_retrieve.return_value = mock_retrieve_agent

        mock_generate_plan.return_value = {"focuses": [{"id": "q_1", "focus": "f1"}]}

        mock_generate_agent = AsyncMock()
        mock_generate_agent.process.return_value = {"success": True, "question": {}}
        mock_create_generate.return_value = mock_generate_agent

        mock_analyzer = AsyncMock()
        mock_analyzer.process.return_value = {"relevance": "high", "kb_coverage": "good"}
        mock_create_analyzer.return_value = mock_analyzer

        result = await coordinator.generate_questions_custom({"knowledge_point": "kp1"}, 1)

        assert result["success"] is True
        assert result["completed"] == 1


@pytest.mark.asyncio
async def test_generate_question_plan(coordinator: AgentCoordinator):
    """Test the _generate_question_plan method.

    Args:
        coordinator (AgentCoordinator): The AgentCoordinator instance.
    """
    with patch(
        "src.agents.question.coordinator.llm_complete", new_callable=AsyncMock
    ) as mock_llm_complete:
        mock_llm_complete.return_value = '{"focuses": [{"id": "q_1", "focus": "f1"}]}'

        plan = await coordinator._generate_question_plan({}, "context", 1)

        assert len(plan["focuses"]) == 1
        assert plan["focuses"][0]["id"] == "q_1"
