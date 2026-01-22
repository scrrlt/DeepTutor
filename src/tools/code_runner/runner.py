"""Runner abstraction and LocalProcessRunner implementation (code_runner package)."""

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

                # Prepare resource limit preexec_fn on POSIX if requested
                preexec_fn = None
                if options.cpu_time_limit or options.memory_limit_bytes:
                    try:
                        import resource  # type: ignore

                        def _set_limits() -> None:  # executed in child
                            if options.cpu_time_limit:
                                resource.setrlimit(
                                    resource.RLIMIT_CPU,
                                    (
                                        options.cpu_time_limit,
                                        options.cpu_time_limit,
                                    ),
                                )
                            if options.memory_limit_bytes:
                                # RLIMIT_AS limits address space (approx. memory usage)
                                resource.setrlimit(
                                    resource.RLIMIT_AS,
                                    (
                                        options.memory_limit_bytes,
                                        options.memory_limit_bytes,
                                    ),
                                )

                        preexec_fn = _set_limits
                    except Exception:  # pragma: no cover - platform-specific
                        logger.warning(
                            "Resource limiting not available on this platform; ignoring limits"
                        )

                # Launch subprocess using asyncio
                proc = await asyncio.create_subprocess_exec(
                    self.python_executable,
                    str(code_file),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=str(work_dir),
                    env=env,
                    preexec_fn=preexec_fn,
                )

                try:
                    stdout_bytes, stderr_bytes = await asyncio.wait_for(
                        proc.communicate(), timeout=options.timeout
                    )
                except TimeoutError:
                    # Kill the process and return timeout result
                    try:
                        proc.kill()
                    except Exception:
                        logger.debug("Failed to kill timed-out process")
                    await proc.wait()

                    elapsed_ms = options.timeout * 1000
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
                elapsed_ms = (time.time() - start) * 1000

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

                # Collect artifacts and build ArtifactInfo records
                from .artifacts import collect_artifacts

                artifacts, artifact_paths = collect_artifacts(work_dir)

                return RunResult(
                    stdout=stdout,
                    stderr=stderr,
                    artifacts=artifacts,
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
