"""
Tests for the MainSolver.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.solve.main_solver import MainSolver


@pytest.mark.asyncio
async def test_main_solver_default_initialization():
    """Test that the MainSolver can be initialized with default parameters."""
    with patch(
        "src.agents.solve.main_solver.load_config_with_main_async",
        new_callable=AsyncMock,
    ) as mock_load_config:
        mock_load_config.return_value = {
            "solve": {
                "agents": {},
            },
            "paths": {},
            "logging": {},
            "tools": {},
            "system": {"language": "en"},
        }

        with patch(
            "src.services.llm.config.get_llm_config_async",
            new_callable=AsyncMock,
        ) as mock_get_llm_config:
            mock_get_llm_config.return_value = MagicMock(
                api_key="test_api_key", base_url="http://localhost:1234"
            )

            solver = MainSolver()
            await solver.ainit()

            assert solver.config is not None
            assert solver.api_key == "test_api_key"
            assert solver.base_url == "http://localhost:1234"
            assert solver.logger is not None
            assert solver.monitor is not None
            assert solver.token_tracker is not None


@pytest.mark.asyncio
async def test_main_solver_custom_config_initialization():
    """Test that the MainSolver can be initialized with a custom configuration file."""
    with patch(
        "src.agents.solve.main_solver.load_config_with_main_async",
        new_callable=AsyncMock,
    ) as mock_load_config:
        mock_load_config.return_value = {
            "solve": {
                "agents": {},
            },
            "paths": {},
            "logging": {},
            "tools": {},
            "system": {
                "language": "en",
                "output_base_dir": "/tmp/test_output",
            },
        }

        with patch(
            "src.services.llm.config.get_llm_config_async",
            new_callable=AsyncMock,
        ) as mock_get_llm_config:
            mock_get_llm_config.return_value = MagicMock(
                api_key="test_api_key", base_url="http://localhost:1234"
            )

            solver = MainSolver(output_base_dir="/tmp/test_output")
            await solver.ainit()

            assert solver.config is not None
            assert solver.config["system"]["output_base_dir"] == "/tmp/test_output"


@pytest.mark.asyncio
async def test_main_solver_ainit_invalid_config():
    """Test that ainit raises a ValueError if the configuration is invalid."""
    with patch(
        "src.agents.solve.main_solver.load_config_with_main_async",
        new_callable=AsyncMock,
    ) as mock_load_config:
        mock_load_config.return_value = {
            "solve": {
                "agents": {"investigate_agent": {"max_iterations": "not_an_integer"}}
            }
        }

        with patch(
            "src.services.llm.config.get_llm_config_async",
            new_callable=AsyncMock,
        ) as mock_get_llm_config:
            mock_get_llm_config.return_value = MagicMock(
                api_key="test_api_key", base_url="http://localhost:1234"
            )

            with pytest.raises(ValueError):
                solver = MainSolver()
                await solver.ainit()


@pytest.mark.asyncio
async def test_main_solver_ainit_missing_api_key():
    """Test that ainit raises a ValueError if the API key is not set."""
    with patch(
        "src.agents.solve.main_solver.load_config_with_main_async",
        new_callable=AsyncMock,
    ) as mock_load_config:
        mock_load_config.return_value = {
            "solve": {
                "agents": {},
            },
            "paths": {},
            "logging": {},
            "tools": {},
            "system": {"language": "en"},
        }

        with patch(
            "src.services.llm.config.get_llm_config_async",
            new_callable=AsyncMock,
        ) as mock_get_llm_config:
            mock_get_llm_config.return_value = MagicMock(
                api_key=None, base_url="https://api.openai.com/v1"
            )

            with pytest.raises(ValueError, match="API key not set"):
                solver = MainSolver()
                await solver.ainit()


@pytest.mark.asyncio
async def test_main_solver_solve_workflow():
    """Test the overall workflow of the MainSolver's solve method."""
    with patch(
        "src.agents.solve.main_solver.load_config_with_main_async",
        new_callable=AsyncMock,
    ) as mock_load_config:
        mock_load_config.return_value = {
            "solve": {
                "agents": {
                    "investigate_agent": {"max_iterations": 1},
                    "precision_answer_agent": {"enabled": True},
                }
            },
            "paths": {},
            "logging": {},
            "tools": {},
            "system": {"language": "en"},
        }

        with patch(
            "src.services.llm.config.get_llm_config_async",
            new_callable=AsyncMock,
        ) as mock_get_llm_config:
            mock_get_llm_config.return_value = MagicMock(
                api_key="test_api_key", base_url="http://localhost:1234"
            )

            solver = MainSolver()
            await solver.ainit()

            with patch.object(
                solver, "_run_dual_loop_pipeline", new_callable=AsyncMock
            ) as mock_run_pipeline:
                mock_run_pipeline.return_value = {
                    "final_answer": "This is the final answer."
                }

                question = "What is the meaning of life?"
                result = await solver.solve(question)

                mock_run_pipeline.assert_called_once()
                assert result["final_answer"] == "This is the final answer."


@pytest.mark.asyncio
async def test_main_solver_dual_loop_pipeline():
    """Test the dual-loop pipeline to ensure agents are called in the correct order."""
    with patch(
        "src.agents.solve.main_solver.load_config_with_main_async",
        new_callable=AsyncMock,
    ) as mock_load_config:
        mock_load_config.return_value = {
            "solve": {
                "agents": {
                    "investigate_agent": {"max_iterations": 1},
                    "precision_answer_agent": {"enabled": True},
                }
            },
            "paths": {},
            "logging": {},
            "tools": {},
            "system": {"language": "en"},
        }

    with patch(
        "src.services.llm.config.get_llm_config_async",
        new_callable=AsyncMock,
    ) as mock_get_llm_config:
        mock_get_llm_config.return_value = MagicMock(
            api_key="test_api_key", base_url="http://localhost:1234"
        )

        solver = MainSolver()
        await solver.ainit()

        # Mock agents
        solver.investigate_agent = AsyncMock()
        solver.note_agent = AsyncMock()
        solver.manager_agent = AsyncMock()
        solver.solve_agent = AsyncMock()
        solver.tool_agent = AsyncMock()
        solver.response_agent = AsyncMock()
        solver.precision_answer_agent = AsyncMock()

        # Mock agent return values
        solver.investigate_agent.process.return_value = {
            "should_stop": True,
            "knowledge_item_ids": ["1"],
        }
        solver.note_agent.process.return_value = {
            "success": True,
            "processed_items": 1,
        }
        solver.manager_agent.process.return_value = {"num_steps": 1}
        solver.solve_agent.process.return_value = {"finish_requested": True}
        solver.response_agent.process.return_value = {"step_response": "response"}
        solver.precision_answer_agent.process.return_value = {
            "needs_precision": True,
            "precision_answer": "precise answer",
        }

        # Mock memory objects
        with (
            patch(
                "src.agents.solve.main_solver.InvestigateMemory"
            ) as mock_investigate_memory,
            patch("src.agents.solve.main_solver.SolveMemory") as mock_solve_memory,
            patch("src.agents.solve.main_solver.CitationMemory"),
        ):
            mock_investigate_memory.load_or_create.return_value = MagicMock(
                knowledge_chain=[]
            )

            mock_step = MagicMock()
            mock_step.status = "waiting_response"
            mock_solve_memory.load_or_create.return_value = MagicMock(
                solve_chains=[mock_step]
            )

            question = "Test question"
            output_dir = "/tmp/test_output"

            await solver._run_dual_loop_pipeline(question, output_dir)

            solver.investigate_agent.process.assert_called_once()
            solver.note_agent.process.assert_called_once()
            solver.manager_agent.process.assert_called_once()
            solver.solve_agent.process.assert_called_once()
            solver.response_agent.process.assert_called_once()
            solver.precision_answer_agent.process.assert_called_once()


@pytest.mark.asyncio
async def test_main_solver_solve_exception_handling():
    """Test that the solve method correctly handles exceptions."""
    with patch(
        "src.agents.solve.main_solver.load_config_with_main_async",
        new_callable=AsyncMock,
    ) as mock_load_config:
        mock_load_config.return_value = {
            "solve": {"agents": {}},
            "paths": {},
            "logging": {},
            "tools": {},
            "system": {"language": "en"},
        }

        with patch(
            "src.services.llm.config.get_llm_config_async",
            new_callable=AsyncMock,
        ) as mock_get_llm_config:
            mock_get_llm_config.return_value = MagicMock(
                api_key="test_api_key", base_url="http://localhost:1234"
            )

            solver = MainSolver()
            await solver.ainit()
            solver.logger = MagicMock()

            with patch.object(
                solver, "_run_dual_loop_pipeline", new_callable=AsyncMock
            ) as mock_run_pipeline:
                mock_run_pipeline.side_effect = Exception("Test Exception")

                with pytest.raises(Exception, match="Test Exception"):
                    await solver.solve("test question")

                assert solver.logger.remove_task_log_handlers.called
                assert solver.logger.shutdown.called
