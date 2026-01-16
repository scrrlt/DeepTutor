#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests for the Logger.
"""

import pytest
from unittest.mock import patch, MagicMock
import logging
from pathlib import Path

from src.logging.logger import Logger, ConsoleFormatter, FileFormatter, get_logger, reset_logger


@pytest.fixture
def logger(tmp_path):
    """
    Provides a Logger instance with a temporary log directory.
    """
    reset_logger()
    log_dir = tmp_path / "logs"
    logger = Logger("TestLogger", log_dir=log_dir, console_output=False)
    yield logger
    logger.shutdown()


def test_logger_initialization(logger: Logger, tmp_path):
    """
    Tests that the Logger can be initialized correctly.
    """
    assert logger.name == "TestLogger"
    assert (tmp_path / "logs").exists()


def test_logging_levels(logger: Logger):
    """
    Tests that the Logger correctly logs messages at different levels.
    """
    with patch.object(logger.logger, 'log') as mock_log:
        logger.debug("debug message")
        mock_log.assert_called_with(logging.DEBUG, "debug message", extra=MagicMock(), exc_info=False, stack_info=False, stacklevel=1)
        
        logger.info("info message")
        mock_log.assert_called_with(logging.INFO, "info message", extra=MagicMock(), exc_info=False, stack_info=False, stacklevel=1)
        
        logger.warning("warning message")
        mock_log.assert_called_with(logging.WARNING, "warning message", extra=MagicMock(), exc_info=False, stack_info=False, stacklevel=1)

        logger.error("error message")
        mock_log.assert_called_with(logging.ERROR, "error message", extra=MagicMock(), exc_info=False, stack_info=False, stacklevel=1)

        logger.critical("critical message")
        mock_log.assert_called_with(logging.CRITICAL, "critical message", extra=MagicMock(), exc_info=False, stack_info=False, stacklevel=1)


def test_formatters():
    """
    Tests that the formatters correctly format the log messages.
    """
    record = logging.LogRecord("test", logging.INFO, "test", 1, "test message", None, None)
    record.module_name = "TestModule"
    
    console_formatter = ConsoleFormatter()
    file_formatter = FileFormatter()

    # Disable colors for predictable output
    console_formatter.use_colors = False
    
    console_output = console_formatter.format(record)
    file_output = file_formatter.format(record)

    assert "[TestModule]" in console_output
    assert "test message" in console_output
    
    assert "[INFO    ]" in file_output
    assert "[TestModule    ]" in file_output
    assert "test message" in file_output


def test_get_logger(tmp_path):
    """
    Tests that the get_logger function correctly creates and retrieves logger instances.
    """
    reset_logger()
    logger1 = get_logger("MyLogger", log_dir=str(tmp_path))
    logger2 = get_logger("MyLogger", log_dir=str(tmp_path))
    assert logger1 is logger2
    logger1.shutdown()


def test_task_handlers(logger: Logger, tmp_path):
    """
    Tests that task handlers are added and removed correctly.
    """
    task_log_file = tmp_path / "task.log"
    logger.add_task_log_handler(str(task_log_file))
    assert len(logger._task_handlers) == 1
    
    logger.info("test task message")
    
    logger.remove_task_log_handlers()
    assert len(logger._task_handlers) == 0

    assert task_log_file.exists()
    assert "test task message" in task_log_file.read_text()


def test_convenience_methods(logger: Logger):
    """
    Tests that the convenience methods work correctly.
    """
    with patch.object(logger, '_log') as mock_log:
        logger.success("success message")
        mock_log.assert_called_with(logging.INFO, "success message", symbol='✓', display_level='SUCCESS')
        
        logger.progress("progress message")
        mock_log.assert_called_with(logging.INFO, "progress message", symbol='→')

        logger.stage("Test Stage", "start")
        mock_log.assert_called_with(logging.INFO, "Test Stage started", symbol='▶', display_level='INFO')
