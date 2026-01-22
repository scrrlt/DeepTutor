"""Compatibility config exports.

Some components and tests historically imported LLM settings from
`src.config.config`. The hardened configuration model lives in
`src.services.llm.config`.

This module provides a stable import path.
"""

from __future__ import annotations

from src.services.llm.config import (
    LLMConfig,
    clear_llm_config_cache,
    get_llm_config,
)

__all__ = [
    "LLMConfig",
    "get_llm_config",
    "clear_llm_config_cache",
]
