#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests for the config loader.
"""

import pytest
from unittest.mock import patch, mock_open
import yaml

from src.services.config.loader import (
    load_config_with_main,
    get_path_from_config,
    parse_language,
    get_agent_params,
)


@patch("builtins.open", new_callable=mock_open)
@patch("src.services.config.loader.Path.exists")
def test_load_config_with_main(mock_exists, mock_file):
    """
    Tests that the load_config_with_main function correctly merges the main and module configurations.
    """
    mock_exists.return_value = True
    
    main_config = {"system": {"language": "en"}, "paths": {"log_dir": "/logs"}}
    module_config = {"system": {"language": "fr"}}

    def mock_safe_load(f):
        if "main" in f.name:
            return main_config
        return module_config

    with patch('yaml.safe_load', side_effect=mock_safe_load):
        config = load_config_with_main("module_config.yaml")

        assert config["system"]["language"] == "fr"
        assert config["paths"]["log_dir"] == "/logs"


def test_get_path_from_config():
    """
    Tests that the get_path_from_config function correctly retrieves the path from the configuration.
    """
    config = {"paths": {"log_dir": "/logs"}, "system": {"workspace": "/workspace"}}
    
    assert get_path_from_config(config, "log_dir") == "/logs"
    assert get_path_from_config(config, "workspace") == "/workspace"
    assert get_path_from_config(config, "non_existent", default="/default") == "/default"


@pytest.mark.parametrize("lang_input, expected", [
    ("en", "en"),
    ("english", "en"),
    ("English", "en"),
    ("zh", "zh"),
    ("chinese", "zh"),
    ("Chinese", "zh"),
    (None, "zh"),
    ("fr", "zh"), # default
])
def test_parse_language(lang_input, expected):
    """
    Tests that the parse_language function correctly parses the language.
    """
    assert parse_language(lang_input) == expected


@patch("builtins.open", new_callable=mock_open)
@patch("src.services.config.loader.Path.exists")
def test_get_agent_params(mock_exists, mock_file):
    """
    Tests that the get_agent_params function correctly retrieves the agent parameters.
    """
    mock_exists.return_value = True
    agents_config = {
        "guide": {"temperature": 0.6, "max_tokens": 2048},
        "solve": {"temperature": 0.7, "max_tokens": 4096},
    }

    with patch('yaml.safe_load', return_value=agents_config):
        guide_params = get_agent_params("guide")
        assert guide_params["temperature"] == 0.6
        assert guide_params["max_tokens"] == 2048

        solve_params = get_agent_params("solve")
        assert solve_params["temperature"] == 0.7
        assert solve_params["max_tokens"] == 4096

        default_params = get_agent_params("non_existent")
        assert default_params["temperature"] == 0.5
        assert default_params["max_tokens"] == 4096
