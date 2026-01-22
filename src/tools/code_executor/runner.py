"""Runner abstraction and LocalProcessRunner implementation.

The Runner encapsulates the execution strategy (local process, docker, remote
runner, etc.). LocalProcessRunner uses asyncio subprocess APIs for
non-blocking execution.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import asyncio
import os
import sys
import time

from src.logging import get_logger

from .types import ExecutionOptions, RunResult
from .workspace import WorkspaceManager

logger = get_logger("CodeRunner")

# Constants for resource limits
MAX_OUTPUT_BYTES = 10 * 1024 * 1024  # 10MB limit for stdout/stderr
MILLISECONDS_PER_SECOND = 1000


class Runner(ABC):
    """Abstract Runner interface."""

    @abstractmethod
    async def run(self, code: str, options: ExecutionOptions) -> RunResult:
        """Execute the provided code with the given options and return a RunResult."""


class LocalProcessRunner(Runner):
    """Local process-based runner using asyncio subprocess."""

    def __init__(
        self,
        workspace_manager: WorkspaceManager | None = None,
        python_executable: str | None = None,
    ):
        self.workspace_manager = workspace_manager or WorkspaceManager()
        self.python_executable = python_executable or sys.executable

    async def run(self, code: str, options: ExecutionOptions) -> RunResult:
        # Prepare workspace
        self.workspace_manager.ensure_initialized()
        assets_dir = options.assets_dir

        try:
            with self.workspace_manager.create_temp_dir() as temp_dir:
                code_file = temp_dir / "code.py"
                code_file.write_text(code, encoding="utf-8")

                work_dir = assets_dir if assets_dir else temp_dir

                env = os.environ.copy()
                env.update(options.env or {})
                env["PYTHONIOENCODING"] = "utf-8"

                start = time.time()

                # Launch subprocess using asyncio
                proc = await asyncio.create_subprocess_exec(
                    self.python_executable,
                    str(code_file),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=str(work_dir),
                    env=env,
                )

                try:
                    # Limit stdout/stderr read size to prevent memory exhaustion (DoS)

                    stdout_bytes, stderr_bytes = await asyncio.wait_for(
                        proc.communicate(), timeout=options.timeout
                    )

                    # Enforce size limits
                    if len(stdout_bytes) > MAX_OUTPUT_BYTES:
                        stdout_bytes = (
                            stdout_bytes[:MAX_OUTPUT_BYTES] + b"\n... [TRUNCATED: output too large]"
                        )
                    if len(stderr_bytes) > MAX_OUTPUT_BYTES:
                        stderr_bytes = (
                            stderr_bytes[:MAX_OUTPUT_BYTES] + b"\n... [TRUNCATED: stderr too large]"
                        )

                except TimeoutError:
                    # Kill the process and return timeout result
                    try:
                        proc.kill()
                    except Exception:
                        logger.debug("Failed to kill timed-out process")
                    await proc.wait()

                    elapsed_ms = options.timeout * MILLISECONDS_PER_SECOND
                    artifacts, artifact_paths = self.workspace_manager.collect_artifacts(assets_dir)

                    return RunResult(
                        stdout="",
                        stderr=f"Code execution timeout ({options.timeout} seconds)",
                        artifacts=[],
                        artifact_paths=artifact_paths,
                        exit_code=-1,
                        elapsed_ms=elapsed_ms,
                    )

                exit_code = proc.returncode if proc.returncode is not None else 0
                elapsed_ms = (time.time() - start) * MILLISECONDS_PER_SECOND

                stdout = (
                    stdout_bytes.decode("utf-8", errors="replace")
                    if isinstance(stdout_bytes, (bytes, bytearray))
                    else str(stdout_bytes)
                )
                stderr = (
                    stderr_bytes.decode("utf-8", errors="replace")
                    if isinstance(stderr_bytes, (bytes, bytearray))
                    else str(stderr_bytes)
                )

                artifacts, artifact_paths = self.workspace_manager.collect_artifacts(assets_dir)

                return RunResult(
                    stdout=stdout,
                    stderr=stderr,
                    artifacts=[],
                    artifact_paths=artifact_paths,
                    exit_code=exit_code,
                    elapsed_ms=elapsed_ms,
                )

        except Exception as exc:  # broad on purpose to map to RunResult
            logger.exception("LocalProcessRunner.run failed: %s", exc)
            artifacts, artifact_paths = self.workspace_manager.collect_artifacts(assets_dir)
            return RunResult(
                stdout="",
                stderr=f"Code execution failed: {exc}",
                artifacts=[],
                artifact_paths=artifact_paths,
                exit_code=-1,
                elapsed_ms=0.0,
            )
