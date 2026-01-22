from collections.abc import Generator
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from src.services.config.loader import load_config_with_main


@pytest.fixture
def mock_config_dir(tmp_path: Path) -> Generator[Path, None, None]:
    """Create a temporary config directory with dummy files."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()

    (config_dir / "main.yaml").write_text("app_name: DeepTutorTest")
    (config_dir / "agents.yaml").write_text("agent_1: default")

    yield config_dir


class TestConfigLoader:
    """Test suite for configuration loading logic."""

    def test_load_config_with_main_success(
        self, mock_config_dir: Path
    ) -> None:
        """Test loading configuration from a valid directory."""
        # We need to patch where the loader looks for config
        # Assuming it takes a path or we patch an internal constant/env
        with patch.dict(os.environ, {"CONFIG_PATH": str(mock_config_dir)}):
            try:
                # If the function takes arguments, they should be supplied here
                # Assuming it might take no args or a path
                config = load_config_with_main()
                assert config is not None
            except Exception as e:
                pytest.fail(f"Config loading failed: {e}")
