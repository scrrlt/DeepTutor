from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ExecutionOptions:
    """Options that control a single execution run.

    Args:
        timeout: Timeout in seconds for the execution.
        assets_dir: Optional path where artifacts are placed/collected.
        allowed_imports: Optional list of module names allowed (ImportGuard use).
        env: Optional dict of environment overrides for the process (merged with
            os.environ when starting the process).
    """

    timeout: int = 10
    assets_dir: Path | None = None
    allowed_imports: list[str] | None = None
    env: dict[str, str] | None = None
    # Resource limits (optional): limits are applied in the child process on
    # POSIX platforms via resource.setrlimit. If not supported on the platform,
    # limits will be ignored and a warning logged.
    cpu_time_limit: int | None = None  # seconds of CPU time
    memory_limit_bytes: int | None = None  # max address space in bytes


@dataclass
class ArtifactInfo:
    """Minimal description of an artifact discovered after execution."""

    name: str
    path: str
    size: int = 0
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass
class RunResult:
    """Structured run result returned by Runner implementations."""

    stdout: str = ""
    stderr: str = ""
    artifacts: list[ArtifactInfo] = field(default_factory=list)
    artifact_paths: list[str] = field(default_factory=list)
    exit_code: int = -1
    elapsed_ms: float = 0.0


class ExecutionError(Exception):
    """Execution-level error raised by Runner implementations when needed.

    Attributes:
        message: Human-readable error message.
    """

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message
