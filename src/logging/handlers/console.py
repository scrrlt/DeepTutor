"""
Console Log Handler
===================

Color-coded console output with standard level tags.
"""

from __future__ import annotations

import sys

from .._stdlib_logging import stdlib_logging
from ..logger import ConsoleFormatter


class ConsoleHandler(stdlib_logging.StreamHandler):
    """
    Console handler with color-coded output.
    """

    def __init__(
        self,
        level: int = stdlib_logging.INFO,
        service_prefix: str | None = None,
    ):
        """
        Initialize console handler.

        Args:
            level: Minimum log level to display.
            service_prefix: Optional service layer prefix (e.g., "Backend", "Frontend").
        """
        super().__init__(sys.stdout)
        self.setLevel(level)
        self.setFormatter(ConsoleFormatter(service_prefix=service_prefix))
