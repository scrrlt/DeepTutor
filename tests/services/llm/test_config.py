# -*- coding: utf-8 -*-
"""Tests for LLM configuration helpers."""

from src.services.llm.config import (
    get_token_limit_kwargs,
    uses_max_completion_tokens,
)


def test_uses_max_completion_tokens_for_supported_models() -> None:
    """Ensure supported models use max_completion_tokens."""
    assert uses_max_completion_tokens("o1-mini")
    assert uses_max_completion_tokens("o1")
    assert uses_max_completion_tokens("o3-mini")
    assert uses_max_completion_tokens("gpt-4o-mini")
    assert uses_max_completion_tokens("gpt-4o")
    assert uses_max_completion_tokens("gpt-5")


def test_uses_max_completion_tokens_false_for_legacy_models() -> None:
    """Ensure legacy models keep max_tokens semantics."""
    assert not uses_max_completion_tokens("gpt-3.5-turbo")


def test_get_token_limit_kwargs_switches() -> None:
    assert get_token_limit_kwargs("gpt-4o-mini", 123) == {"max_completion_tokens": 123}
    assert get_token_limit_kwargs("gpt-3.5-turbo", 321) == {"max_tokens": 321}
