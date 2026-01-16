import logging
from pathlib import Path
from unittest.mock import patch

import pytest
from src.logging.logger import Logger, get_logger


class TestLogger:
    """Test suite for the Logger class."""

    def test_logger_initialization(self, tmp_path: Path):
        """Test logger initialization.

        Args:
            tmp_path (Path): Pytest fixture for a temporary directory.
        """
        logger = Logger("TestModule", log_dir=str(tmp_path))
        assert logger.name == "TestModule"
        assert logger.logger.name == "ai_tutor.TestModule"
        assert logger.log_dir == tmp_path

    def test_get_logger_singleton(self, tmp_path: Path):
        """Test that get_logger returns a singleton for the same parameters.

        Args:
            tmp_path (Path): Pytest fixture for a temporary directory.
        """
        logger1 = get_logger("TestModule", log_dir=str(tmp_path))
        logger2 = get_logger("TestModule", log_dir=str(tmp_path))
        assert logger1 is logger2

        logger3 = get_logger("OtherModule", log_dir=str(tmp_path))
        assert logger1 is not logger3

    def test_log_methods(self, tmp_path: Path):
        """Test the different log methods.

        Args:
            tmp_path (Path): Pytest fixture for a temporary directory.
        """
        logger = Logger("TestModule", log_dir=str(tmp_path), console_output=False)

        with patch.object(logger.logger, "log") as mock_log:
            logger.info("Info message")
            mock_log.assert_called_with(logging.INFO, "Info message")

            logger.error("Error message")
            mock_log.assert_called_with(logging.ERROR, "Error message")

            logger.success("Success message")
            mock_log.assert_called_with(
                logging.INFO, "Success message", extra={"display_level": "SUCCESS"}
            )

    def test_task_handlers(self, tmp_path: Path):
        """Test adding and removing task-specific log handlers.

        Args:
            tmp_path (Path): Pytest fixture for a temporary directory.
        """
        logger = Logger("TestTask", log_dir=str(tmp_path))
        task_log_path = tmp_path / "task.log"

        logger.add_task_log_handler(str(task_log_path))
        assert len(logger._task_handlers) == 1, "Task handler should be added."

        logger.info("Task message")

        logger.remove_task_log_handlers()
        assert len(logger._task_handlers) == 0, "Task handler should be removed."

        assert task_log_path.exists(), "Log file should be created."
        content = task_log_path.read_text(encoding="utf-8")
        assert "Task message" in content, "Log file should contain the task message."