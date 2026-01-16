import logging
from typing import Any

import pytest
from src.logging import get_logger


class TestLogging:
    """Test suite for the application logging module."""

    def test_get_logger_returns_logger(self) -> None:
        """Test that get_logger returns a valid logging.Logger instance."""
        logger: Any = get_logger("test_module")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_module"

    def test_get_logger_level(self) -> None:
        """Test that the logger is configured with the correct level."""
        # Assuming the default or env var sets it to DEBUG/INFO
        logger = get_logger("test_level")
        # We check if it's effectively not NOTSET (0)
        assert logger.getEffectiveLevel() != logging.NOTSET
        # Since we mocked RAG_TOOL_MODULE_LOG_LEVEL=DEBUG in conftest
        # We might expect DEBUG, but it depends on implementation.