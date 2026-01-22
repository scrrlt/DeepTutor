from pathlib import Path
from unittest.mock import patch

import pytest

from src.services.config.loader import (
    get_agent_params,
    load_config_with_main,
    parse_language,
)


@pytest.fixture
def mock_config_dir(tmp_path: Path):
    """Create a temporary config directory with dummy files."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()

    (config_dir / "main.yaml").write_text(
        "app_name: DeepTutorTest\nsystem:\n  language: en", encoding="utf-8"
    )
    (config_dir / "agents.yaml").write_text(
        "agent_1: default", encoding="utf-8"
    )

    yield config_dir


class TestConfigLoader:
    """Test suite for configuration loading logic."""

    def test_load_config_with_main_success(self, mock_config_dir: Path):
        """Test loading configuration from a valid directory."""
        # Create a module config
        (mock_config_dir / "test_module.yaml").write_text(
            "module_key: module_value", encoding="utf-8"
        )

        # We need to patch where the loader looks for config
        project_root = mock_config_dir.parent
        config = load_config_with_main(
            "test_module.yaml", project_root=project_root
        )

        assert config is not None
        assert config["app_name"] == "DeepTutorTest"
        assert config["module_key"] == "module_value"

    def test_parse_language(self):
        """Test language parsing."""
        assert parse_language("zh") == "zh"
        assert parse_language("Chinese") == "zh"
        assert parse_language("ZH") == "zh"

        assert parse_language("en") == "en"
        assert parse_language("English") == "en"
        assert parse_language("EN") == "en"

        assert parse_language(None) == "zh"
        assert parse_language("unknown") == "zh"

    def test_get_agent_params_defaults(self):
        """Test getting agent params when file doesn't exist or agent not found."""
        # Mock PROJECT_ROOT to point to a place without agents.yaml
        with patch(
            "src.services.config.loader.PROJECT_ROOT", Path("/non/existent")
        ):
            params = get_agent_params("unknown_agent")
            assert params["temperature"] == 0.5
            assert params["max_tokens"] == 4096

    def test_get_agent_params_loaded(self, mock_config_dir: Path):
        """Test getting agent params from file."""
        agents_yaml = """
        test_agent:
            temperature: 0.7
            max_tokens: 2048
        """
        (mock_config_dir / "agents.yaml").write_text(
            agents_yaml, encoding="utf-8"
        )

        project_root = mock_config_dir.parent
        with patch("src.services.config.loader.PROJECT_ROOT", project_root):
            params = get_agent_params("test_agent")
            assert params["temperature"] == 0.7
            assert params["max_tokens"] == 2048
