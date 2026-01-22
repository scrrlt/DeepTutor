from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.research.research_pipeline import ResearchPipeline


@pytest.fixture
def research_pipeline() -> ResearchPipeline:
    """Provides a ResearchPipeline instance with mocked dependencies.

    Yields:
        ResearchPipeline: An instance of the ResearchPipeline.
    """
    with patch("src.agents.research.research_pipeline.get_logger"):
        config = {
            "system": {
                "output_base_dir": "/tmp/test_cache",
                "reports_dir": "/tmp/test_reports",
            },
            "planning": {
                "rephrase": {"enabled": True},
                "decompose": {"mode": "manual", "initial_subtopics": 2},
            },
            "researching": {
                "execution_mode": "series",
                "max_parallel_topics": 2,
            },
            "reporting": {},
            "queue": {},
            "rag": {"kb_name": "test_kb"},
            "paths": {"user_log_dir": "/tmp/logs"},
        }
        pipeline = ResearchPipeline(
            config=config, api_key="test_key", base_url="http://localhost:1234"
        )
        yield pipeline


@pytest.mark.asyncio
async def test_research_pipeline_initialization(
    research_pipeline: ResearchPipeline,
):
    """Test that the ResearchPipeline can be initialized correctly.

    Args:
        research_pipeline (ResearchPipeline): The ResearchPipeline instance.
    """
    assert research_pipeline.config is not None
    assert research_pipeline.api_key == "test_key"
    assert research_pipeline.base_url == "http://localhost:1234"
    assert research_pipeline.logger is not None
    assert research_pipeline.agents is not None
    assert (
        len(research_pipeline.agents) == 6
    )  # rephrase, decompose, manager, research, note, reporting


@pytest.mark.asyncio
async def test_research_pipeline_run_workflow(
    research_pipeline: ResearchPipeline,
):
    """Test the overall workflow of the ResearchPipeline's run method.

    Args:
        research_pipeline (ResearchPipeline): The ResearchPipeline instance.
    """
    with (
        patch.object(
            research_pipeline, "_phase1_planning", new_callable=AsyncMock
        ) as mock_planning,
        patch.object(
            research_pipeline, "_phase2_researching", new_callable=AsyncMock
        ) as mock_researching,
        patch.object(
            research_pipeline, "_phase3_reporting", new_callable=AsyncMock
        ) as mock_reporting,
        patch("builtins.open", new_callable=MagicMock),
    ):
        mock_planning.return_value = "optimized_topic"
        mock_reporting.return_value = {
            "report": "final_report",
            "word_count": 100,
        }

        topic = "Test Topic"
        result = await research_pipeline.run(topic)

        mock_planning.assert_called_once_with(topic)
        mock_researching.assert_called_once()
        mock_reporting.assert_called_once_with("optimized_topic")

        assert result["topic"] == topic
        assert "final_report_path" in result


@pytest.mark.asyncio
async def test_phase1_planning(research_pipeline: ResearchPipeline):
    """Test the _phase1_planning method.

    Args:
        research_pipeline (ResearchPipeline): The ResearchPipeline instance.
    """
    research_pipeline.agents["rephrase"].process = AsyncMock(
        return_value={"topic": "optimized topic"}
    )
    research_pipeline.agents["decompose"].process = AsyncMock(
        return_value={"sub_topics": [{"title": "sub1"}, {"title": "sub2"}]}
    )

    topic = "Test Topic"
    optimized_topic = await research_pipeline._phase1_planning(topic)

    assert optimized_topic == "optimized topic"
    assert research_pipeline.queue.get_statistics()["total_blocks"] == 2
    research_pipeline.agents["rephrase"].process.assert_called_once()
    research_pipeline.agents["decompose"].process.assert_called_once()


@pytest.mark.asyncio
async def test_phase2_researching_series(research_pipeline: ResearchPipeline):
    """Test the _phase2_researching method in series mode.

    Args:
        research_pipeline (ResearchPipeline): The ResearchPipeline instance.
    """
    # Populate the queue
    research_pipeline.queue.add_block("sub1")
    research_pipeline.queue.add_block("sub2")

    research_pipeline.agents["research"].process = AsyncMock(return_value={"iterations": 1})

    await research_pipeline._phase2_researching()

    assert research_pipeline.queue.get_statistics()["completed"] == 2
    assert research_pipeline.agents["research"].process.call_count == 2


@pytest.mark.asyncio
async def test_phase2_researching_parallel(
    research_pipeline: ResearchPipeline,
):
    """Test the _phase2_researching method in parallel mode.

    Args:
        research_pipeline (ResearchPipeline): The ResearchPipeline instance.
    """
    research_pipeline.config["researching"]["execution_mode"] = "parallel"

    # Populate the queue
    research_pipeline.queue.add_block("sub1")
    research_pipeline.queue.add_block("sub2")

    research_pipeline.agents["research"].process = AsyncMock(return_value={"iterations": 1})

    await research_pipeline._phase2_researching()

    assert research_pipeline.queue.get_statistics()["completed"] == 2
    assert research_pipeline.agents["research"].process.call_count == 2


@pytest.mark.asyncio
async def test_phase3_reporting(research_pipeline: ResearchPipeline):
    """Test the _phase3_reporting method.

    Args:
        research_pipeline (ResearchPipeline): The ResearchPipeline instance.
    """
    research_pipeline.agents["reporting"].process = AsyncMock(
        return_value={
            "report": "final_report",
            "word_count": 100,
            "sections": 1,
            "citations": 0,
        }
    )

    topic = "Test Topic"
    result = await research_pipeline._phase3_reporting(topic)

    assert result["report"] == "final_report"
    assert result["word_count"] == 100
    research_pipeline.agents["reporting"].process.assert_called_once()
