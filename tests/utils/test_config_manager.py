#!/usr/bin/env python

"""
Tests for the ConfigManager.
"""

import pytest

from src.utils.config_manager import ConfigError, ConfigManager


@pytest.fixture
def config_manager(tmp_path):
    """
    Provides a ConfigManager instance with a temporary project root.
    """
    ConfigManager.reset_for_tests()
    manager = ConfigManager(project_root=tmp_path)
    (tmp_path / "config").mkdir()
    yield manager
    ConfigManager.reset_for_tests()


def test_config_manager_singleton(tmp_path):
    """
    Tests that the ConfigManager is a singleton.
    """
    cm1 = ConfigManager(project_root=tmp_path)
    cm2 = ConfigManager(project_root=tmp_path)
    assert cm1 is cm2


def test_load_config(config_manager: ConfigManager, tmp_path):
    """
    Tests that the load_config method correctly loads and validates the configuration.
    """
    config_path = tmp_path / "config" / "main.yaml"
    config_path.write_text('version: "1.0"\nsystem:\n  language: "en"')

    config = config_manager.load_config()
    assert config["system"]["language"] == "en"

    # Test caching
    config_path.write_text('version: "1.0"\nsystem:\n  language: "fr"')
    config = config_manager.load_config()
    assert config["system"]["language"] == "en"  # Should still be cached

    config = config_manager.load_config(force_reload=True)
    assert config["system"]["language"] == "fr"


def test_save_config(config_manager: ConfigManager):
    """
    Tests that the save_config method correctly saves the configuration.
    """
    new_config = {"system": {"language": "de"}}
    config_manager.save_config(new_config)

    config = config_manager.load_config()
    assert config["system"]["language"] == "de"


def test_load_invalid_config(config_manager: ConfigManager, tmp_path):
    """
    Tests that load_config returns an empty dict for invalid config.
    """
    config_path = tmp_path / "config" / "main.yaml"
    config_path.write_text("system:\n  language: 123")  # Invalid type

    with pytest.raises(ConfigError):
        config_manager._validate_and_migrate({"system": {"language": 123}})

    config = config_manager.load_config(force_reload=True)
    assert config == {}


def test_save_invalid_config(config_manager: ConfigManager):
    """
    Tests that save_config rejects invalid config.
    """
    assert not config_manager.save_config({"system": {"language": 123}})


def test_env_info(config_manager: ConfigManager, tmp_path):
    """
    Tests that get_env_info and validate_required_env work correctly.
    """
    (tmp_path / ".env").write_text("LLM_MODEL=test_model")
    (tmp_path / ".env.local").write_text("LLM_API_KEY=test_key")

    env_info = config_manager.get_env_info()
    assert env_info["model"] == "test_model"

    validation_result = config_manager.validate_required_env(
        ["LLM_API_KEY", "MISSING_KEY"]
    )
    assert "MISSING_KEY" in validation_result["missing"]
    assert "LLM_API_KEY" not in validation_result["missing"]
