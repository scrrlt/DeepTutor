"""Artifact collection utilities for code execution workspaces.

Provides functions to discover artifacts produced by executed code, sanitize
filenames, and produce structured ArtifactInfo objects.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from .types import ArtifactInfo


def _compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _sanitize_name(name: str) -> str:
    # Minimal sanitization: remove path separators and leading dots
    return name.replace("..", "_").lstrip("/\\")


def collect_artifacts(work_dir: Path) -> tuple[list[ArtifactInfo], list[str]]:
    """Collect artifacts from a directory.

    Args:
        work_dir: Directory to scan for artifact files.

    Returns:
        (artifact_infos, artifact_paths)
    """
    artifacts: list[ArtifactInfo] = []
    artifact_paths: list[str] = []

    if not work_dir or not work_dir.exists():
        return artifacts, artifact_paths

    for p in sorted(work_dir.iterdir()):
        if p.is_file() and p.name != ".gitkeep":
            try:
                size = p.stat().st_size
                sha = _compute_sha256(p)
            except Exception:
                size = 0
                sha = ""

            name = _sanitize_name(p.name)
            artifacts.append(
                ArtifactInfo(
                    name=name,
                    path=str(p.resolve()),
                    size=size,
                    metadata={"sha256": sha},
                )
            )
            artifact_paths.append(str(p.resolve()))

    return artifacts, artifact_paths
