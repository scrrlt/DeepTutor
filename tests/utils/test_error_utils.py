# -*- coding: utf-8 -*-
"""Tests for error utilities (formatting and user-friendly messages)."""

from src.utils.error_utils import format_exception_message, user_friendly_message
from src.services.llm.exceptions import LLMConfigError, LLMAuthenticationError


def test_format_exception_message_extracts_json_block() -> None:
    msg = "Request failed: 400 {\"error\": {\"message\": \"Invalid API key\", \"type\": \"invalid_request_error\"}}"
    exc = Exception(msg)

    formatted = format_exception_message(exc)

    assert "Invalid API key" in formatted
    assert "invalid_request_error" in formatted


def test_user_friendly_message_for_llm_config_error() -> None:
    exc = LLMConfigError("Model is required")
    friendly = user_friendly_message(exc)

    assert "LLM configuration error" in friendly
    assert "Model is required" in friendly


def test_user_friendly_message_for_authentication_error() -> None:
    exc = LLMAuthenticationError("API key missing", provider="openai")
    friendly = user_friendly_message(exc)

    assert "Authentication failed" in friendly or "Authentication failed" in friendly
    assert "API key missing" in friendly
