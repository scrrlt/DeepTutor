"""Shared fixtures for service tests."""

from pathlib import Path

from dotenv import load_dotenv
import pytest


def _load_env_file(path: Path) -> None:
    """Load environment variables from a dotenv file if it exists."""
    if path.exists():
        load_dotenv(path, override=False)


@pytest.fixture(scope="session", autouse=True)
def load_service_env_keys() -> None:
    """Load API keys from local dotenv files for LLM and embedding tests."""
    project_root = Path(__file__).resolve().parents[2]
    _load_env_file(project_root / ".env.local")
    _load_env_file(project_root / "web" / ".env.local")


def pytest_addoption(parser: pytest.Parser) -> None:
    """
    Register shared pytest command line options for service tests.

    Args:
        parser: Pytest argument parser.

    Returns:
        None.

    Raises:
        None.
    """
    parser.addoption(
        "--pipeline",
        action="store",
        default="llamaindex",
        help="Pipeline to test",
    )
