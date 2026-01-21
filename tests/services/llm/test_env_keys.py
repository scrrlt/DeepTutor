# -*- coding: utf-8 -*-
"""Tests for LLM API key environment configuration."""

from __future__ import annotations

import os
import re
from typing import Iterable, Mapping

import pytest

KEY_PATTERNS: Mapping[str, str] = {
    "ANTHROPIC_API_KEY": r"^sk-ant-[A-Za-z0-9_-]{10,}$",
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

PROVIDER_KEY_MAP: Mapping[str, list[str]] = {
    "openai": ["OPENAI_API_KEY"],
    "azure": ["AZURE_API_KEY"],
    "azure_openai": ["AZURE_API_KEY"],
    "anthropic": [
        "ANTHROPIC_API_KEY",
        "CLAUDE_API_KEY",  # Legacy alias for backward compatibility
    ],
    "gemini": ["GEMINI_API_KEY"],
    "vertex": ["VERTEX_API_KEY"],
    "cohere": ["COHERE_API_KEY"],
    "jina": ["JINA_API_KEY"],
    "ollama": ["OLLAMA_API_KEY"],
    "together": ["TOGETHER_API_KEY"],
    "serper": ["SERPER_API_KEY"],
}


def _missing_keys(keys: Iterable[str]) -> list[str]:
    """
    Identify which environment variable names are missing or contain only whitespace.
    
    Parameters:
        keys (Iterable[str]): Environment variable names to check.
    
    Returns:
        list[str]: Names from `keys` whose environment value is None or empty after trimming.
    """
    missing: list[str] = []
    for key in keys:
        value = os.getenv(key)
        if value is None or not value.strip():
            missing.append(key)
    return missing


def _invalid_key_formats(keys: Mapping[str, str]) -> list[str]:
    """
    Return the names of environment variables whose current values do not match the provided regex patterns.
    
    Parameters:
        keys (Mapping[str, str]): Mapping of environment variable names to regex pattern strings.
    
    Returns:
        list[str]: List of variable names for which a non-empty environment value exists and does not match its pattern. Missing or empty values are ignored.
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


def _get_keys_for_binding(binding: str) -> list[str]:
    """
    Return the list of environment variable names required for the given provider binding.
    
    Parameters:
        binding (str): Provider binding name (case-insensitive); must be a supported provider key.
    
    Returns:
        list[str]: Environment variable names required for the binding.
    
    Raises:
        ValueError: If `binding` is empty or not one of the supported provider keys.
    """
    normalized = binding.strip().lower()
    if not normalized or normalized not in PROVIDER_KEY_MAP:
        valid = ", ".join(sorted(PROVIDER_KEY_MAP.keys()))
        raise ValueError(f"Unknown LLM provider binding {binding!r}. Expected one of: {valid}.")
    return PROVIDER_KEY_MAP[normalized]


def test_required_llm_api_keys_present() -> None:
    """
    Verifies that all required LLM API key environment variables for the configured binding are present.
    
    Skips the test if the environment variable ENFORCE_LLM_KEYS is not set to "1". The active binding is taken from LLM_BINDING (defaults to "openai"); the test checks the keys required for that binding and fails if any are missing.
    
    Raises:
        AssertionError: If one or more required environment variables are missing; the assertion message lists the missing keys.
    """
    if os.getenv("ENFORCE_LLM_KEYS") != "1":
        pytest.skip("ENFORCE_LLM_KEYS is not set")

    binding = os.getenv("LLM_BINDING", "openai")
    keys_to_check = _get_keys_for_binding(binding)
    missing = _missing_keys(keys_to_check)

    assert not missing, "Missing required API keys in environment: " f"{', '.join(missing)}."


def test_llm_api_keys_have_expected_format() -> None:
    """
    Check that configured LLM provider API key environment variables match their expected formats.
    
    Skips the test unless ENFORCE_LLM_KEYS is "1". Reads LLM_BINDING (defaults to "openai") to determine which keys to validate, then asserts all present keys match their regex patterns; on failure lists keys with invalid formats.
    """
    if os.getenv("ENFORCE_LLM_KEYS") != "1":
        pytest.skip("ENFORCE_LLM_KEYS is not set")

    binding = os.getenv("LLM_BINDING", "openai")
    keys_to_check = _get_keys_for_binding(binding)
    patterns = {key: KEY_PATTERNS[key] for key in keys_to_check if key in KEY_PATTERNS}
    invalid = _invalid_key_formats(patterns)

    assert not invalid, "API keys did not match expected formats: " f"{', '.join(invalid)}."