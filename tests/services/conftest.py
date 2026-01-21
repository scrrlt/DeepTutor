# -*- coding: utf-8 -*-
"""Shared fixtures for service tests."""

from pathlib import Path

from dotenv import load_dotenv
import pytest


def _load_env_file(path: Path) -> None:
    """
    Load environment variables from a dotenv file at the given path if that file exists.
    
    Parameters:
        path (Path): Filesystem path to a dotenv file. Existing environment variables are preserved; variables in the file will not override existing values.
    """
    if path.exists():
        load_dotenv(path, override=False)


@pytest.fixture(scope="session", autouse=True)
def load_service_env_keys() -> None:
    """
    Load environment variables for service tests from local dotenv files.
    
    If present, loads variables from `.env.local` in the project root and from `web/.env.local`, making those values available to tests.
    """
    project_root = Path(__file__).resolve().parents[2]
    _load_env_file(project_root / ".env.local")
    _load_env_file(project_root / "web" / ".env.local")


def pytest_addoption(parser: pytest.Parser) -> None:
    """
    Register the shared pytest command-line option used to select the service pipeline under test.
    
    Adds a `--pipeline` option with default value `"llamaindex"`.
    """
    parser.addoption(
        "--pipeline",
        action="store",
        default="llamaindex",
        help="Pipeline to test",
    )