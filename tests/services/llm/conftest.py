# -*- coding: utf-8 -*-
"""Shared fixtures for LLM service tests."""

from pathlib import Path

import pytest
from dotenv import load_dotenv


def _load_env_file(path: Path) -> None:
    """
    Load environment variables from a dotenv file if it exists.

    Args:
        path: Path to the dotenv file.
    """
    if path.exists():
        load_dotenv(path, override=False)


@pytest.fixture(scope="session", autouse=True)
def load_llm_env_keys() -> None:
    """
    Load API keys from local dotenv files for LLM tests.
    """
    project_root = Path(__file__).resolve().parents[3]
    _load_env_file(project_root / ".env.local")
    _load_env_file(project_root / "web" / ".env.local")
