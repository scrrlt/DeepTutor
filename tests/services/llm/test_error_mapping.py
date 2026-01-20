# -*- coding: utf-8 -*-
"""Tests for LLM error mapping helpers."""

from src.services.llm.error_mapping import map_error
from src.services.llm.exceptions import ProviderContextWindowError


def test_context_window_error_mapping() -> None:
    """Ensure context length errors map to ProviderContextWindowError."""
    mapped = map_error(
        Exception("maximum context length exceeded"),
        provider="openai",
    )
    assert isinstance(mapped, ProviderContextWindowError)
