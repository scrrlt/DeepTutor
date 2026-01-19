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
    assert (
        sanitize_url("https://api.openai.com/v1/chat/completions")
        == "https://api.openai.com/v1"
    )


def test_build_chat_url_anthropic() -> None:
    url = build_chat_url("https://api.anthropic.com/v1", binding="anthropic")
    assert url.endswith("/messages")


def test_build_chat_url_azure_api_version() -> None:
    url = build_chat_url(
        "https://example.azure.com/openai/deployments/test",
        api_version="2024-02-01",
    )
    assert url.endswith("/chat/completions?api-version=2024-02-01")


def test_clean_thinking_tags() -> None:
    content = "<think>skip</think>answer"
    assert (
        clean_thinking_tags(
            content, binding="deepseek", model="deepseek-reasoner"
        )
        == "answer"
    )
    assert (
        clean_thinking_tags(content, binding="openai", model="gpt-4")
        == content
    )


def test_extract_response_content_falls_back() -> None:
    assert extract_response_content({"reasoning": "insight"}) == "insight"
