from src.services.config.unified_config import UnifiedConfigManager, ConfigType


def test_get_config_default_returns_default_for_llm() -> None:
    """Ensure get_config returns the default configuration for LLM."""
    # Reset any singleton state to make the test isolated
    UnifiedConfigManager._instance = None

    manager = UnifiedConfigManager()
    cfg = manager.get_config(ConfigType.LLM, "default")

    assert isinstance(cfg, dict)
    assert cfg["id"] == "default"


def test_get_active_llm_config_uses_current_manager() -> None:
    """Ensure get_active_llm_config calls get_active_config on the current manager."""
    from unittest.mock import patch

    # Reset singleton and create an instance
    UnifiedConfigManager._instance = None
    manager = UnifiedConfigManager()

    with patch.object(manager, "get_active_config") as mock_get:
        from src.services.config.unified_config import get_active_llm_config, ConfigType

        get_active_llm_config()
        mock_get.assert_called_with(ConfigType.LLM)
