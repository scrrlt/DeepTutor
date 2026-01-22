"""Code Runner package.

Provides a small abstraction around code execution runners (local process,
Docker, remote runners). This package is intentionally lightweight and
import-safe in test environments.
"""

from .runner import LocalProcessRunner, Runner
from .types import ArtifactInfo, ExecutionOptions, RunResult

__all__ = [
    "Runner",
    "LocalProcessRunner",
    "ExecutionOptions",
    "RunResult",
    "ArtifactInfo",
]
