import os

import pytest

from src.services.llm.config import (
    LLMConfig,
    LLMConfigError,
    clear_llm_config_cache,
    get_llm_config,
    reload_config,
)
from src.services.llm.model_rules import (
    get_token_limit_kwargs,
    uses_max_completion_tokens,
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """
    Clean up environment variables and reset the singleton before/after each test.
    This prevents test pollution.
    """
    # Clear relevant env vars
    for key in os.environ:
        if key.startswith("LLM_"):
            monkeypatch.delenv(key)

    # Reset singleton without validating env
    clear_llm_config_cache()
    yield
    clear_llm_config_cache()


def test_config_validation_failure():
    """Test that LLMConfig can be instantiated with defaults and get_llm_config returns a config."""
    cfg = LLMConfig()
    assert isinstance(cfg, LLMConfig)

    cfg2 = get_llm_config()
    assert isinstance(cfg2, LLMConfig)


def test_config_load_from_env(monkeypatch):
    """Test loading basic configuration from environment variables."""
    monkeypatch.setenv("LLM_MODEL", "gpt-4-test")
    monkeypatch.setenv("LLM_API_KEY", "sk-test-key")
    monkeypatch.setenv("LLM_TEMPERATURE", "0.5")

    config = get_llm_config()

    assert config.model == "gpt-4-test"
    assert config.get_api_key() == "sk-test-key"
    assert config.temperature == 0.5
    assert config.binding == "openai"  # Default
    assert config.max_tokens == 4096  # Default


def test_effective_url_presets(monkeypatch):
    """Test that effective_url correctly resolves presets for known bindings."""
    monkeypatch.setenv("LLM_MODEL", "llama3")
    monkeypatch.setenv("LLM_BINDING", "ollama")

    config = LLMConfig()
    assert config.effective_url == "http://localhost:11434/v1"


def test_effective_url_override(monkeypatch):
    """Test that an explicit base_url overrides the binding default."""
    monkeypatch.setenv("LLM_MODEL", "llama3")
    monkeypatch.setenv("LLM_BINDING", "ollama")
    monkeypatch.setenv("LLM_BASE_URL", "http://my-remote-server:11434")

    config = LLMConfig()
    assert config.effective_url == "http://my-remote-server:11434"


def test_effective_url_unknown_binding(monkeypatch):
    """Test that unknown bindings without a base_url raise an error."""
    monkeypatch.setenv("LLM_MODEL", "custom-model")
    monkeypatch.setenv("LLM_BINDING", "unknown_provider")

    config = LLMConfig()

    # Accessing the computed field triggers the logic
    with pytest.raises(LLMConfigError, match="requires explicit base_url"):
        _ = config.effective_url


def test_alias_provider_name():
    """Test that 'provider_name' in the constructor maps to 'binding'."""
    config = LLMConfig(model="gpt-4", provider_name="anthropic")
    assert config.binding == "anthropic"
    assert config.provider_name == "anthropic"


@pytest.mark.parametrize(
    "model, expected_param",
    [
        ("gpt-4o", "max_completion_tokens"),
        ("GPT-4o-2024-08-06", "max_completion_tokens"),
        ("o1-preview", "max_completion_tokens"),
        ("o1-mini", "max_completion_tokens"),
        ("o3-mini", "max_completion_tokens"),
        ("gpt-3.5-turbo", "max_tokens"),
        ("claude-3-5-sonnet", "max_tokens"),
        ("deepseek-coder", "max_tokens"),
    ],
)
def test_token_param_resolution(model, expected_param):
    """Token-limit parameter selection lives in model_rules (not config)."""
    kwargs = get_token_limit_kwargs(model, 123)
    assert list(kwargs.keys()) == [expected_param]


def test_compatibility_shims():
    """Test the standalone compatibility functions."""
    # uses_max_completion_tokens
    assert uses_max_completion_tokens("gpt-4o") is True
    assert uses_max_completion_tokens("o1-preview") is True
    assert uses_max_completion_tokens("gpt-4") is False

    # get_token_limit_kwargs
    kwargs_4o = get_token_limit_kwargs("gpt-4o", 100)
    assert kwargs_4o == {"max_completion_tokens": 100}

    kwargs_35 = get_token_limit_kwargs("gpt-3.5", 100)
    assert kwargs_35 == {"max_tokens": 100}


def test_singleton_behavior(monkeypatch):
    """Verify get_llm_config returns the same instance until reload."""
    monkeypatch.setenv("LLM_MODEL", "initial")

    cfg1 = get_llm_config()
    cfg2 = get_llm_config()
    assert cfg1 is cfg2
    assert cfg1.model == "initial"

    # Change env, should NOT affect cached config
    monkeypatch.setenv("LLM_MODEL", "changed")
    cfg3 = get_llm_config()
    assert cfg3.model == "initial"

    # Reload should pick up change
    cfg4 = reload_config()
    assert cfg4.model == "changed"
    assert cfg4 is not cfg1
