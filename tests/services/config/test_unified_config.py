import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from src.services.config.unified_config import (
    ConfigType,
    UnifiedConfigManager,
    get_active_llm_config,
    get_config_manager,
)


class TestUnifiedConfigManager:
    """Test suite for the UnifiedConfigManager class."""

    @pytest.fixture
    def mock_settings_dir(self, tmp_path: Path):
        """Mock the settings directory."""
        with patch("src.services.config.unified_config.SETTINGS_DIR", tmp_path):
            yield tmp_path

    @pytest.fixture
    def config_manager(self, mock_settings_dir: Path):
        """Fixture to provide a UnifiedConfigManager instance with reset singleton."""
        # Reset singleton to ensure fresh state
        UnifiedConfigManager.reset_for_tests()
        manager = UnifiedConfigManager()
        return manager

    def test_singleton_pattern(self):
        """Test that UnifiedConfigManager follows singleton pattern."""
        UnifiedConfigManager.reset_for_tests()
        m1 = UnifiedConfigManager()
        m2 = UnifiedConfigManager()
        assert m1 is m2
        assert get_config_manager() is m1

    def test_build_default_config_llm(self, config_manager: UnifiedConfigManager):
        """Test building default LLM config from environment."""
        env_vars = {
            "LLM_BINDING": "openai",
            "LLM_MODEL": "gpt-4-test",
            "LLM_API_KEY": "sk-test",
            "LLM_HOST": "https://api.test.com",
        }
        with patch.dict(os.environ, env_vars):
            config = config_manager._build_default_config(ConfigType.LLM)
            assert config["id"] == "default"
            assert config["provider"] == "openai"
            assert config["model"] == "gpt-4-test"
            assert config["base_url"] == "https://api.test.com"
            assert config["is_default"] is True
            # API key should be hidden in default config view
            assert config["api_key"] == "***"

    def test_list_configs(self, config_manager: UnifiedConfigManager, mock_settings_dir: Path):
        """Test listing configurations."""
        # Create a dummy config file
        config_file = mock_settings_dir / "llm_configs.json"
        data = {
            "configs": [{"id": "custom1", "name": "Custom 1", "provider": "anthropic"}],
            "active_id": "custom1",
        }
        config_file.write_text(json.dumps(data), encoding="utf-8")

        configs = config_manager.list_configs(ConfigType.LLM)
        assert len(configs) == 2  # default + custom1

        default_cfg = next(c for c in configs if c["id"] == "default")
        custom_cfg = next(c for c in configs if c["id"] == "custom1")

        assert default_cfg["is_active"] is False
        assert custom_cfg["is_active"] is True

    def test_add_config(self, config_manager: UnifiedConfigManager):
        """Test adding a new configuration."""
        new_config = {
            "name": "New Config",
            "provider": "ollama",
            "base_url": "http://localhost:11434",
        }
        result = config_manager.add_config(ConfigType.LLM, new_config)

        assert result["id"] is not None
        assert result["provider"] == "ollama"
        assert result["is_default"] is False

        # Verify it's in the list
        configs = config_manager.list_configs(ConfigType.LLM)
        assert len(configs) == 2  # default + new

    def test_update_config(self, config_manager: UnifiedConfigManager):
        """Test updating an existing configuration."""
        # Add a config first
        added = config_manager.add_config(
            ConfigType.LLM, {"name": "Old Name", "provider": "openai"}
        )
        cid = added["id"]

        # Update it
        updated = config_manager.update_config(ConfigType.LLM, cid, {"name": "New Name"})
        assert updated is not None
        assert updated["name"] == "New Name"
        assert updated["id"] == cid

        # Verify persistence
        configs = config_manager.list_configs(ConfigType.LLM)
        stored = next(c for c in configs if c["id"] == cid)
        assert stored["name"] == "New Name"

    def test_update_default_config_fail(self, config_manager: UnifiedConfigManager):
        """Test that updating default config fails."""
        result = config_manager.update_config(ConfigType.LLM, "default", {"name": "New Name"})
        assert result is None

    def test_delete_config(self, config_manager: UnifiedConfigManager):
        """Test deleting a configuration."""
        added = config_manager.add_config(ConfigType.LLM, {"name": "To Delete"})
        cid = added["id"]

        # Delete
        assert config_manager.delete_config(ConfigType.LLM, cid) is True

        # Verify it's gone
        configs = config_manager.list_configs(ConfigType.LLM)
        assert len(configs) == 1  # only default remains

        # Try deleting again
        assert config_manager.delete_config(ConfigType.LLM, cid) is False

    def test_delete_default_config_fail(self, config_manager: UnifiedConfigManager):
        """Test that deleting default config fails."""
        assert config_manager.delete_config(ConfigType.LLM, "default") is False

    def test_set_active_config(self, config_manager: UnifiedConfigManager):
        """Test setting the active configuration."""
        added = config_manager.add_config(ConfigType.LLM, {"name": "Active Candidate"})
        cid = added["id"]

        assert config_manager.set_active_config(ConfigType.LLM, cid) is True

        configs = config_manager.list_configs(ConfigType.LLM)
        active = next(c for c in configs if c["is_active"])
        assert active["id"] == cid

    def test_get_active_llm_config_helper(self, config_manager: UnifiedConfigManager):
        """Test the helper function get_active_llm_config."""
        # Ensure helper uses the manager
        with patch.object(config_manager, "get_active_config") as mock_get:
            get_active_llm_config()
            mock_get.assert_called_with(ConfigType.LLM)
