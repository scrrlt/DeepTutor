# -*- coding: utf-8 -*-
"""Shared fixtures for embedding service tests."""

from pathlib import Path
import pytest
from dotenv import load_dotenv


def _load_env_file(path: Path) -> None:
    """Load environment variables from a dotenv file."""
    if path.exists():
        load_dotenv(path, override=False)


@pytest.fixture(scope="session", autouse=True)
def load_embedding_env_keys() -> None:
    """Load API keys from local dotenv files for embedding tests."""
    project_root = Path(__file__).resolve().parents[3]
    _load_env_file(project_root / ".env.local")
    _load_env_file(project_root / "web" / ".env.local")
