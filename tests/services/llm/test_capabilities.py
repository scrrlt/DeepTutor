"""Tests for capability resolution."""

from src.services.llm.capabilities import (
    get_capability,
    has_thinking_tags,
    requires_api_version,
    supports_response_format,
    supports_streaming,
    supports_tools,
    system_in_messages,
)


def test_supports_response_format_respects_overrides() -> None:
    assert not supports_response_format("deepseek", "deepseek-reasoner")


def test_capability_defaults_for_unknown_provider() -> None:
    assert supports_streaming("unknown-provider")
    assert not supports_tools("unknown-provider")


def test_system_in_messages_false_for_anthropic() -> None:
    """Ensure Anthropic keeps system prompts out of messages."""
    assert not system_in_messages("anthropic")


def test_requires_api_version_true_for_azure_openai() -> None:
    """Ensure Azure OpenAI requires an api_version parameter."""
    assert requires_api_version("azure_openai")


def test_has_thinking_tags_for_qwen() -> None:
    assert has_thinking_tags("openai", "qwen-2.5")


def test_get_capability_explicit_default() -> None:
    assert get_capability("unknown-provider", "nonexistent", default=True)
