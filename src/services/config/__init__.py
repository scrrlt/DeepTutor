"""
Configuration Service
=====================

Unified configuration loading for all DeepTutor modules.

Usage:
    from src.services.config import load_config_with_main, PROJECT_ROOT

    # Load module configuration
    config = load_config_with_main("solve_config.yaml")

    # Get agent parameters
    params = get_agent_params("guide")
"""

from src.utils.config_manager import ConfigManager

# Create a global instance for backward compatibility
_config_manager = ConfigManager()

# Backward compatibility functions
PROJECT_ROOT = _config_manager.project_root


def load_config_with_main(config_file: str, project_root=None):
    """Backward compatibility wrapper."""
    return _config_manager.load_config_with_module(config_file)


def get_path_from_config(config, path_key, default=None):
    """Backward compatibility wrapper."""
    return _config_manager.get_path_from_config(config, path_key, default)


def parse_language(language):
    """Backward compatibility wrapper."""
    return _config_manager.parse_language(language)


def get_agent_params(module_name):
    """Backward compatibility wrapper."""
    return _config_manager.get_agent_params(module_name)


def _deep_merge(base, override):
    """Backward compatibility - simple deep merge."""
    if not isinstance(override, dict):
        return base
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


__all__ = [
    "PROJECT_ROOT",
    "load_config_with_main",
    "get_path_from_config",
    "parse_language",
    "get_agent_params",
    "_deep_merge",
]
