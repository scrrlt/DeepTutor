"""
Console Log Handler
===================

Color-coded console output with standard level tags.
"""

import sys

from .._stdlib_logging import stdlib_logging


class ConsoleFormatter(stdlib_logging.Formatter):
    """
    Clean console formatter with colors and symbols.
    Format: [Module]    Symbol Message
    """

    # ANSI color codes
    COLORS = {
        "DEBUG": "\033[90m",  # Gray
        "INFO": "\033[37m",  # White
        "SUCCESS": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    # Symbols for different log types
    SYMBOLS = {
        "DEBUG": "·",
        "INFO": "●",
        "SUCCESS": "✓",
        "WARNING": "⚠",
        "ERROR": "✗",
        "CRITICAL": "✗",
    }

    def format(self, record: stdlib_logging.LogRecord) -> str:
        # Get module name (padded to 12 chars for alignment)
        module = getattr(record, "module_name", record.name)
        module_padded = f"[{module}]".ljust(14)

        # Get symbol (can be overridden via record.symbol)
        symbol = getattr(
            record, "symbol", self.SYMBOLS.get(record.levelname, "●")
        )

        # Get color
        level = getattr(record, "display_level", record.levelname)
        color = self.COLORS.get(level, self.COLORS["INFO"])

        # Format message
        message = record.getMessage()

        # Build output: [Module]    ● Message
        return f"{self.DIM}{module_padded}{self.RESET} {color}{symbol}{self.RESET} {message}"
from typing import Optional

# Import ConsoleFormatter from the main logger module to avoid duplication
from ..logger import ConsoleFormatter


class ConsoleHandler(stdlib_logging.StreamHandler):
    """
    Console handler with color-coded output.
    """

    def __init__(self, level: int = stdlib_logging.INFO):
    def __init__(self, level: int = logging.INFO, service_prefix: Optional[str] = None):
        """
        Initialize console handler.

        Args:
            level: Minimum log level to display
            service_prefix: Optional service layer prefix (e.g., "Backend", "Frontend")
        """
        super().__init__(sys.stdout)
        self.setLevel(level)
        self.setFormatter(ConsoleFormatter(service_prefix=service_prefix))
