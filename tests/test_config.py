import os
from unittest.mock import patch

import pytest
from src.services.llm.config import get_llm_config


class TestLLMConfig:
    """Test suite for LLM configuration retrieval."""

    def test_get_llm_config_defaults(self) -> None:
        """Test retrieving LLM config returns expected structure."""
        config = get_llm_config()
        assert isinstance(config, dict)
        # Check for keys we mocked in conftest.py
        assert config.get("model") == "gpt-4o"
        assert config.get("binding") == "openai"

    def test_get_llm_config_override(self) -> None:
        """Test that environment variables override defaults."""
        with patch.dict(os.environ, {"LLM_MODEL": "claude-3-opus"}):
            # We might need to reload the module or call the function again
            # depending on if it caches. Assuming it reads env on call.
            config = get_llm_config()
            assert config.get("model") == "claude-3-opus"