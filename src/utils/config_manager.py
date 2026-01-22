from __future__ import annotations

import logging
import os
from pathlib import Path
import tempfile
from threading import Lock
from typing import Any

from dotenv import dotenv_values, load_dotenv
from pydantic import ValidationError
import yaml

from src.config.defaults import DEFAULTS
from src.config.schema import AppConfig, migrate_config
from src.core.errors import ConfigError

logger = logging.getLogger(__name__)


class ConfigManager:
    """
    Thread-safe manager for reading and writing configuration files.

    Primarily manages config/main.yaml and reads .env.
    """

    _instance: ConfigManager | None = None
    _config_cache: dict[str, Any] = {}
    _lock = Lock()

    def __new__(cls, project_root: Path | None = None) -> ConfigManager:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, project_root: Path | None = None) -> None:
        if getattr(self, "_initialized", False):
            return

        self.project_root = project_root or Path(__file__).parent.parent.parent
        self.config_path = self.project_root / "config" / "main.yaml"
        self._config_cache = {}
        self._last_mtime = 0.0
        self._initialized = True

        skip_dotenv = os.getenv("CONFIG_MANAGER_SKIP_DOTENV", "").lower()
        if skip_dotenv not in {"1", "true", "yes"}:
            self._initialize_env()

    def _initialize_env(self) -> None:
        """Initialize process environment from layered .env files."""
        load_dotenv(dotenv_path=self.project_root / ".env", override=False)
        load_dotenv(dotenv_path=self.project_root / ".env.local", override=True)

    def _load_env_file(self, path: Path) -> dict[str, str]:
        """Load a .env file and return non-None values as strings."""
        if not path.exists():
            return {}
        return {k: str(v) for k, v in dotenv_values(path).items() if v is not None}

    def _read_yaml(self) -> dict[str, Any]:
        """Read the main YAML configuration file safely."""
        if not self.config_path.exists():
            return {}
        with open(self.config_path, encoding="utf-8") as file:
            return yaml.safe_load(file) or {}

    def _deep_update(self, target: dict[str, Any], source: dict[str, Any]) -> None:
        for key, value in source.items():
            if isinstance(value, dict) and isinstance(target.get(key), dict):
                self._deep_update(target[key], value)
            else:
                target[key] = value

    def _validate_and_migrate(self, raw: dict[str, Any]) -> dict[str, Any]:
        merged: dict[str, Any] = {}
        self._deep_update(merged, DEFAULTS)
        self._deep_update(merged, raw)
        migrated = migrate_config(merged)
        try:
            return AppConfig(**migrated).dict()
        except ValidationError as exc:
            raise ConfigError(
                "Config validation failed",
                context={"errors": exc.errors()},
            ) from exc

    def load_config(self, force_reload: bool = False) -> dict[str, Any]:
        """
        Load configuration from main.yaml.

        Uses caching based on file modification time and validates against schema.
        """
        with self._lock:
            if not self.config_path.exists():
                logger.info("Config not found at %s", self.config_path)
                self._config_cache = {}
                self._last_mtime = 0
                return {}

            if not self._config_cache or force_reload:
                try:
                    raw = self._read_yaml()
                    validated = self._validate_and_migrate(raw)
                    self._config_cache = validated
                    self._last_mtime = self.config_path.stat().st_mtime
                except ConfigError as exc:
                    logger.error(
                        "%s",
                        exc,
                        extra={"context": getattr(exc, "context", {})},
                    )
                    return {}
                except Exception as exc:
                    logger.exception("Error loading config: %s", exc)
                    return {}

            return (
                yaml.safe_load(yaml.safe_dump(self._config_cache, sort_keys=False))
                or {}
            )

    def save_config(self, config: dict[str, Any]) -> bool:
        """
        Save configuration to main.yaml.

        Deep-merges provided config with existing one; writes atomically.
        Rejects invalid configs per schema.
        """
        try:
            with self._lock:
                current = self.load_config(force_reload=True)
                self._deep_update(current, config)
                validated = self._validate_and_migrate(current)

                self.config_path.parent.mkdir(parents=True, exist_ok=True)
                yaml_str = yaml.safe_dump(
                    validated,
                    default_flow_style=False,
                    allow_unicode=True,
                    sort_keys=False,
                )

                fd, tmp_path = tempfile.mkstemp(
                    prefix="main.yaml.",
                    dir=str(self.config_path.parent),
                )
                try:
                    with os.fdopen(fd, "w", encoding="utf-8") as tmp:
                        tmp.write(yaml_str)
                        tmp.flush()
                        os.fsync(tmp.fileno())

                    backup_path = self.config_path.with_suffix(".yaml.bak")
                    if self.config_path.exists():
                        try:
                            os.replace(self.config_path, backup_path)
                        except Exception:
                            logger.debug("Backup replace failed; continuing.")

                    os.replace(tmp_path, self.config_path)
                    self._config_cache = validated
                    self._last_mtime = self.config_path.stat().st_mtime
                    return True
                finally:
                    if os.path.exists(tmp_path):
                        try:
                            os.remove(tmp_path)
                        except Exception as exc:
                            logger.warning(
                                "Failed to remove temp file %s: %s",
                                tmp_path,
                                exc,
                            )
        except ConfigError as exc:
            logger.error(
                "Refusing to save invalid config: %s",
                exc,
                extra={"context": getattr(exc, "context", {})},
            )
            return False
        except Exception as exc:
            logger.exception("Error saving config: %s", exc)
            return False

    def get_env_info(self) -> dict[str, str]:
        """Read relevant environment variables.

        Reads layered .env files plus the process environment.

        Args:
            None.
        Returns:
            Mapping of environment variable values.
        Raises:
            None.
        """
        env_path = self.project_root / ".env"
        local_path = self.project_root / ".env.local"
        parsed_env = self._load_env_file(env_path)
        parsed_env.update(self._load_env_file(local_path))

        def _get(key: str, default: str = "") -> str:
            return str(parsed_env.get(key) or os.environ.get(key, default))

        return {
            "model": _get(
                "LLM_MODEL",
                DEFAULTS.get("llm", {}).get("model", "Pro/Flash"),
            ),
        }

    def validate_required_env(self, keys: list[str]) -> dict[str, list[str]]:
        """
        Validate that required environment variables are set.

        Args:
            keys: List of environment variable keys to validate.

        Returns:
            Dictionary containing missing keys.
        """
        env_path = self.project_root / ".env"
        local_path = self.project_root / ".env.local"
        parsed_env = self._load_env_file(env_path)
        parsed_env.update(self._load_env_file(local_path))

        missing = [
            key for key in keys if not (parsed_env.get(key) or os.environ.get(key))
        ]
        if missing:
            logger.warning("Missing required env keys", extra={"missing": missing})
        return {"missing": missing}
