"""
Compatibility helpers for configuration access.

Re-exports configuration utilities from :mod:`src.services.config` to
support legacy imports.
"""

from src.services.config import (
    PROJECT_ROOT,
    get_agent_params,
    get_path_from_config,
    load_config_with_main,
    parse_language,
)

__all__ = [
    "PROJECT_ROOT",
    "load_config_with_main",
    "get_path_from_config",
    "parse_language",
    "get_agent_params",
]
