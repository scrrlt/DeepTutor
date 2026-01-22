# -*- coding: utf-8 -*-
"""Tests for LLM utility helpers."""

from src.services.llm.utils import (
    build_chat_url,
    clean_thinking_tags,
    extract_response_content,
    is_local_llm_server,
    sanitize_url,
)


def test_is_local_llm_server() -> None:
    assert is_local_llm_server("http://localhost:1234/v1")
    assert not is_local_llm_server("https://api.openai.com/v1")


def test_sanitize_url_adds_protocol_and_v1() -> None:
    assert sanitize_url("localhost:1234") == "http://localhost:1234/v1"


def test_sanitize_url_strips_chat_suffix() -> None:
    assert sanitize_url("https://api.openai.com/v1/chat/completions") == "https://api.openai.com/v1"


def test_build_chat_url_anthropic() -> None:
    url = build_chat_url("https://api.anthropic.com/v1", binding="anthropic")
    assert url.endswith("/messages")


def test_build_chat_url_azure_api_version() -> None:
    url = build_chat_url(
        "https://example.azure.com/openai/deployments/test",
        api_version="2024-02-01",
    )
    assert url.endswith("/chat/completions?api-version=2024-02-01")


def test_build_chat_url_openai_default() -> None:
    url = build_chat_url("https://api.openai.com/v1", binding="openai")
    assert url.endswith("/chat/completions")
    assert "?" not in url


def test_clean_thinking_tags() -> None:
    content = "<think>skip</think>answer"
    assert clean_thinking_tags(content, binding="deepseek", model="deepseek-reasoner") == "answer"
    assert clean_thinking_tags(content, binding="openai", model="gpt-4") == content


def test_extract_response_content_uses_content_field() -> None:
    assert extract_response_content({"content": "main content"}) == "main content"


def test_extract_response_content_priority() -> None:
    """Verify reasoning field does not override primary content."""
    data = {"content": "Primary Answer", "reasoning": "Thinking Process"}
    # Should return content, ignoring reasoning
    assert extract_response_content(data) == "Primary Answer"


def test_extract_response_content_falls_back_to_reasoning() -> None:
    assert extract_response_content({"reasoning": "insight"}) == "insight"


def test_extract_response_content_reasoning_fallback() -> None:
    """Verify fallback to reasoning if content is missing or empty."""
    assert (
        extract_response_content({"content": "", "reasoning": "Only Thinking"}) == "Only Thinking"
    )
    assert (
        extract_response_content({"content": None, "reasoning": "Only Thinking"}) == "Only Thinking"
    )


def test_extract_response_content_passes_through_strings() -> None:
    assert extract_response_content("direct string") == "direct string"


def test_extract_response_content_handles_empty_and_none() -> None:
    assert extract_response_content(None) == ""
    assert extract_response_content({}) == ""
