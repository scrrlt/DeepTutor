# -*- coding: utf-8 -*-
"""Tests for LLM API key environment configuration."""

from __future__ import annotations

import os
import re
from typing import Iterable, Mapping


KEY_PATTERNS: Mapping[str, str] = {
    "AZURE_API_KEY": r"^[A-Za-z0-9_-]{20,}$",
    "GEMINI_API_KEY": r"^AIza[0-9A-Za-z_-]{10,}$",
    "VERTEX_API_KEY": r"^AQ\.[A-Za-z0-9_-]{10,}$",
    "OPENAI_API_KEY": r"^sk-(proj-|svcacct-)?[A-Za-z0-9_-]{10,}$",
    "CLAUDE_API_KEY": r"^sk-ant-[A-Za-z0-9_-]{10,}$",
    "TOGETHER_API_KEY": r"^tgp_v1_[A-Za-z0-9]{10,}$",
    "SERPER_API_KEY": r"^[A-Fa-f0-9]{20,}$",
    "COHERE_API_KEY": r"^AQ[A-Za-z0-9_-]{10,}$",
    "JINA_API_KEY": r"^jina_[A-Za-z0-9_-]{10,}$",
    "OLLAMA_API_KEY": r"^[A-Za-z0-9.-]{10,}$",
}


def _missing_keys(keys: Iterable[str]) -> list[str]:
    """
    Collect missing environment variables.

    Args:
        keys: Environment variable names to validate.

    Returns:
        List of missing variable names.
    """
    missing: list[str] = []
    for key in keys:
        value = os.getenv(key)
        if value is None or not value.strip():
            missing.append(key)
    return missing


def _invalid_key_formats(keys: Mapping[str, str]) -> list[str]:
    """
    Collect environment variables that fail expected format checks.

    Args:
        keys: Mapping of variable names to regex patterns.

    Returns:
        List of variable names that fail format validation.
    """
    invalid: list[str] = []
    for name, pattern in keys.items():
        value = os.getenv(name)
        if not value:
            continue
        normalized = value.strip()
        if normalized and not re.match(pattern, normalized):
            invalid.append(name)
    return invalid


def test_required_llm_api_keys_present() -> None:
    missing = _missing_keys(KEY_PATTERNS.keys())

    assert not missing, (
        "Missing required API keys in environment: " f"{', '.join(missing)}."
    )


def test_llm_api_keys_have_expected_format() -> None:
    invalid = _invalid_key_formats(KEY_PATTERNS)

    assert not invalid, (
        "API keys did not match expected formats: " f"{', '.join(invalid)}."
    )
