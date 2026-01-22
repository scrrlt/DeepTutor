#!/usr/bin/env python
"""
Config Validator - Configuration validator
Validates the completeness and correctness of config.yaml using Pydantic models.
"""

from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from src.config.constants import PROJECT_ROOT
from src.config.schema import AppConfig
from src.logging import get_logger

logger = get_logger(__name__)


class ConfigValidator:
    """Configuration validator using Pydantic models"""

    def __init__(self):
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def validate(self, config: dict[str, Any]) -> tuple[bool, list[str], list[str]]:
        """
        Validate configuration

        Args:
            config: Configuration dictionary

        Returns:
            (is_valid, errors, warnings)
        """
        self.errors = []
        self.warnings = []

        try:
            # Pydantic handles type checking, range validation (ge, le), and existence checks
            AppConfig.model_validate(config)
        except ValidationError as e:
            for error in e.errors():
                loc = " -> ".join(str(x) for x in error["loc"])
                self.errors.append(f"[{loc}] {error['msg']}")

        # Add warnings for deprecated or non-optimal settings if needed
        if "system" in config and "output_language" in config["system"]:
            self.warnings.append(
                "output_language is deprecated, please use system.language in config/main.yaml"
            )

        is_valid = len(self.errors) == 0
        return is_valid, self.errors, self.warnings


def validate_config_file(config_path: str) -> tuple[bool, list[str], list[str]]:
    """
    Validate configuration file

    Args:
        config_path: Configuration file path

    Returns:
        (is_valid, errors, warnings)
    """
    try:
        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        return False, [f"Configuration file does not exist: {config_path}"], []
    except yaml.YAMLError as e:
        return False, [f"YAML parsing error: {e!s}"], []
    except Exception as e:
        return False, [f"Failed to load configuration file: {e!s}"], []

    validator = ConfigValidator()
    return validator.validate(config)


def print_validation_result(is_valid: bool, errors: list[str], warnings: list[str]):
    """
    Print validation result

    Args:
        is_valid: Whether configuration is valid
        errors: List of errors
        warnings: List of warnings
    """
    logger.info("=" * 60)
    logger.info("Configuration Validation Result")
    logger.info("=" * 60)

    if is_valid:
        logger.info("✓ Configuration validation passed")
    else:
        logger.error("✗ Configuration validation failed")

    logger.info()

    if errors:
        logger.error(f"Errors ({len(errors)}):")
        for i, error in enumerate(errors, 1):
            logger.error(f"  {i}. {error}")
        logger.info()

    if warnings:
        logger.warning(f"Warnings ({len(warnings)}):")
        for i, warning in enumerate(warnings, 1):
            logger.warning(f"  {i}. {warning}")
        logger.info()

    logger.info("=" * 60)


if __name__ == "__main__":
    # Test configuration validation
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
    logger.info("Configuration Validation Test")
=======
    print("Configuration Validation Test")
>>>>>>> cb09a95 (feat: Replace print statements with proper logging)
    logger.info("=" * 60)
=======
    logger.section("Configuration Validation Test")
    logger.info("%s", "=" * 60)
>>>>>>> b97c4c6 (Refactor logging and improve code clarity across multiple modules)
=======
    logger.info("Configuration Validation Test")
    logger.info("=" * 60)
>>>>>>> d73408a (refactor: improve logging consistency and clarity in various modules)

    # Validate config.yaml in current directory
    config_path = PROJECT_ROOT / "config.yaml"

    if config_path.exists():
        is_valid, errors, warnings = validate_config_file(str(config_path))
        print_validation_result(is_valid, errors, warnings)
    else:
        logger.info(f"Configuration file not found: {config_path}")
