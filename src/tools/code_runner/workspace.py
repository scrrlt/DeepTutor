"""Workspace utilities for isolated code execution (code_runner package)."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
import os
from pathlib import Path
import tempfile

# Environment keys (can be overridden in environment)
RUN_CODE_WORKSPACE_ENV = "RUN_CODE_WORKSPACE"
RUN_CODE_ALLOWED_ROOTS_ENV = "RUN_CODE_ALLOWED_ROOTS"
DEFAULT_WORKSPACE_NAME = "run_code_workspace"

# Project root is expected to be set by callers if needed; defaulting to cwd
PROJECT_ROOT = Path.cwd()


class WorkspaceManager:
    """Manages isolated workspace directories for code execution."""

    def __init__(self, project_root: Path | None = None):
        # Resolve project root
        self.project_root = project_root or PROJECT_ROOT

        # Determine workspace directory (priority: env var > default)
        env_path = os.getenv(RUN_CODE_WORKSPACE_ENV)
        if env_path:
            self.base_dir = Path(env_path).expanduser().resolve()
        else:
            # Default workspace is set under user directory
            self.base_dir = (
                self.project_root / "data" / "user" / DEFAULT_WORKSPACE_NAME
            ).resolve()

        # Determine allowed roots (allow project and user data by default)
        self.allowed_roots: list[Path] = [
            self.project_root.resolve(),
            (self.project_root / "data" / "user").resolve(),
        ]

        # Append any additional allowed roots from env var
        extra_roots = os.getenv(RUN_CODE_ALLOWED_ROOTS_ENV)
        if extra_roots:
            for raw in extra_roots.split(os.pathsep):
                raw = raw.strip()
                if raw:
                    p = Path(raw).expanduser()
                    if not p.is_absolute():
                        p = (self.project_root / p).resolve()
                    if p not in self.allowed_roots:
                        self.allowed_roots.append(p)

        # Ensure workspace itself is allowed
        if self.base_dir not in self.allowed_roots:
            self.allowed_roots.append(self.base_dir)

        self._initialized = False

    def initialize(self) -> None:
        """Create base workspace directory if needed."""
        if not self._initialized:
            self.base_dir.mkdir(parents=True, exist_ok=True)
            self._initialized = True

    def ensure_initialized(self) -> None:
        """Ensure the workspace is initialized."""
        if not self._initialized:
            self.initialize()

    @contextmanager
    def create_temp_dir(self) -> Generator[Path, None, None]:
        """Create a TemporaryDirectory inside the workspace base dir."""
        self.ensure_initialized()
        with tempfile.TemporaryDirectory(dir=str(self.base_dir)) as temp_dir:
            yield Path(temp_dir)

    def resolve_assets_dir(self, assets_dir: str | None) -> Path | None:
        """Resolve and validate a user-specified assets directory."""
        if not assets_dir:
            return None
        path = Path(assets_dir).expanduser()
        if not path.is_absolute():
            path = (self.base_dir / path).resolve()
        self._ensure_within_allowed_roots(path)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def collect_artifacts(
        self, assets_dir: Path | None
    ) -> tuple[list[str], list[str]]:
        """Collect artifact filenames and absolute paths from the assets dir."""
        artifacts: list[str] = []
        artifact_paths: list[str] = []
        if not assets_dir or not assets_dir.exists():
            return artifacts, artifact_paths

        for file_path in assets_dir.iterdir():
            if file_path.is_file() and file_path.name != ".gitkeep":
                artifacts.append(str(file_path.relative_to(assets_dir)))
                artifact_paths.append(str(file_path.resolve()))
        return artifacts, artifact_paths

    def _ensure_within_allowed_roots(self, path: Path) -> None:
        """Ensure that the given path is inside one of the allowed roots."""
        resolved_path = path.resolve()
        for root in self.allowed_roots:
            try:
                if hasattr(resolved_path, "is_relative_to"):
                    if resolved_path.is_relative_to(root):
                        return
                else:
                    resolved_str = (
                        str(resolved_path).lower().replace("\\", "/")
                    )
                    root_str = str(root.resolve()).lower().replace("\\", "/")
                    if resolved_str.startswith(root_str):
                        return
            except Exception:
                resolved_str = str(resolved_path).lower().replace("\\", "/")
                root_str = str(root.resolve()).lower().replace("\\", "/")
                if resolved_str.startswith(root_str):
                    return
        allowed = "\n".join(str(r) for r in self.allowed_roots)
        raise ValueError(
            f"Assets directory {resolved_path} must be located under one of the following allowed paths:\n{allowed}"
        )
