import os
from unittest.mock import patch

from src.services.llm.config import LLMConfig, get_llm_config, reload_config


class TestLLMConfig:
    """Test suite for LLM configuration retrieval."""

    def test_get_llm_config_defaults(self) -> None:
        """Test retrieving LLM config returns expected structure."""
        config = get_llm_config()
        assert isinstance(config, LLMConfig)
        # Check for keys we mocked in conftest.py
        assert config.model == "gpt-4o"
        assert config.binding == "openai"

    def test_get_llm_config_override(self) -> None:
        """Test that environment variables override defaults."""
        with patch.dict(os.environ, {"LLM_MODEL": "claude-3-opus"}):
            # Force reload so the env override is picked up
            reload_config()
            config = get_llm_config()
            assert config.model == "claude-3-opus"
            # Reset cache to avoid leaking to other tests
            reload_config()
