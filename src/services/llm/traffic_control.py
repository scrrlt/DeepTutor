"""Traffic control primitives for LLM providers."""
from __future__ import annotations

from typing import Any


class TrafficController:
    """Async context manager that currently performs no rate limiting."""

    def __init__(self, provider_name: str) -> None:
        """Initialize a traffic controller for the given provider."""
        self.provider_name = provider_name

    async def __aenter__(self) -> "TrafficController":
        """Enter the async context (no-op)."""
        return self

    async def __aexit__(
        self,
        exc_type: Any,
        exc: Any,
        tb: Any,
    ) -> None:
        """Exit the async context (no-op)."""
        return None


__all__ = ["TrafficController"]
