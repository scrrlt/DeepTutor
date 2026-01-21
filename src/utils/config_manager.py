import logging
import os
from pathlib import Path
import tempfile
from threading import Lock
from typing import Any

from dotenv import dotenv_values, load_dotenv
from pydantic import ValidationError
import yaml

from ..config.defaults import DEFAULTS

# Use package-relative imports to avoid PYTHONPATH issues
from ..config.schema import AppConfig, migrate_config
from ..core.errors import ConfigError

logger = logging.getLogger(__name__)


class ConfigManager:
    """
    Thread-safe manager for reading and writing configuration files.
    Primarily manages config/main.yaml and reads .env.

    Governance additions:
    - Schema validation via pydantic (AppConfig); invalid configs are rejected.
    - Versioned migrations via migrate_config.
    - Atomic writes with temp file and os.replace; creates main.yaml.bak.
    - Single lock guards mtime read, load, and save.
    - Deterministic YAML dumps; returns deep copies.
    - Layered env: .env, then .env.local (override), then process env.
    """

    _instance: "ConfigManager | None" = None
    _config_cache: dict[str, Any] = {}
    _lock = Lock()

    def __new__(cls, project_root: Path | None = None):
        """
        Create or return the singleton ConfigManager instance.
        
        Initializes the singleton on first call in a thread-safe manner; subsequent calls return the same instance.
        
        Parameters:
            project_root (Path | None): Optional project root path forwarded to the instance initializer on first instantiation.
        
        Returns:
            ConfigManager: The singleton ConfigManager instance.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(ConfigManager, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, project_root: Path | None = None):
        """
        Initialize the ConfigManager singleton instance and prepare file paths, in-memory cache, and layered environment loading.
        
        Parameters:
            project_root (Path | None): Root directory for the project. If None, defaults to the directory three levels above this file. The instance will use this root to locate config/main.yaml.
        
        Description:
            - Sets self.config_path to {project_root}/config/main.yaml.
            - Initializes an in-memory config cache and modification-time tracker.
            - Marks the instance as initialized to enforce singleton behavior.
            - Loads environment variables from .env (without overriding existing env values) and then from .env.local (overriding when present).
        """
        if getattr(self, "_initialized", False):
            return

        self.project_root = project_root or Path(__file__).parent.parent.parent
        self.config_path = self.project_root / "config" / "main.yaml"
        self._config_cache: dict[str, Any] = {}
        self._last_mtime: float = 0.0
        self._initialized = True

        # Layered env loading
        load_dotenv(dotenv_path=self.project_root / ".env", override=False)
        load_dotenv(dotenv_path=self.project_root / ".env.local", override=True)

    def _load_env_file(self, path: Path) -> dict[str, str]:
        """
        Load values from a dotenv file into a mapping.
        
        Only keys with a non-None value are included. If the file does not exist, returns an empty dict.
        
        Returns:
            A dict mapping environment variable names to their values as strings; keys with no value are omitted.
        """
        if not path.exists():
            return {}
        return {k: str(v) for k, v in dotenv_values(path).items() if v is not None}

    def _read_yaml(self) -> dict[str, Any]:
        """
        Load and parse the main YAML configuration file.
        
        Returns:
            dict: Parsed configuration mapping. Returns an empty dict if the config file does not exist or contains no data.
        """
        if not self.config_path.exists():
            return {}
        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def _deep_update(self, target: dict[str, Any], source: dict[str, Any]) -> None:
        """
        Recursively merge values from `source` into `target`, updating `target` in place.
        
        Nested dictionaries are merged rather than replaced; for matching keys where both
        values are dictionaries, their contents are merged recursively. For non-dictionary
        values, or when a key is absent in `target`, the value from `source` overwrites
        or is assigned to `target`.
        
        Parameters:
            target (dict[str, Any]): Dictionary to be updated in place.
            source (dict[str, Any]): Dictionary providing values to merge into `target`.
        """
        for key, value in source.items():
            if isinstance(value, dict) and isinstance(target.get(key), dict):
                self._deep_update(target[key], value)
            else:
                target[key] = value

    def _validate_and_migrate(self, raw: dict[str, Any]) -> dict[str, Any]:
        """
        Validate and migrate a raw configuration dictionary and return a validated config.
        
        The function merges the provided `raw` config with built-in defaults, applies schema migrations, and validates the result against the AppConfig schema.
        
        Parameters:
            raw (dict[str, Any]): Unvalidated configuration data to be merged, migrated, and validated.
        
        Returns:
            dict[str, Any]: The validated and migrated configuration as a plain dictionary.
        
        Raises:
            ConfigError: If schema validation fails; the exception includes validation error details.
        """
        merged: dict[str, Any] = {}
        self._deep_update(merged, DEFAULTS)
        self._deep_update(merged, raw)
        migrated = migrate_config(merged)
        try:
            return AppConfig(**migrated).dict()
        except ValidationError as e:
            raise ConfigError("Config validation failed", details={"errors": e.errors()})

    def load_config(self, force_reload: bool = False) -> dict[str, Any]:
        """
        Load and return the application's configuration from config/main.yaml.
        
        If the file exists, validate and migrate its contents, update the internal cache based on file modification time, and return an immutable deep copy of the validated configuration. If the file is missing, fails validation, or an error occurs while reading, an empty dictionary is returned.
        
        Parameters:
        	force_reload (bool): If True, bypass cached data and reload the file even if it has not changed.
        
        Returns:
        	dict[str, Any]: The validated configuration as a deep-copied dict, or an empty dict on missing file or error.
        """
        with self._lock:
            if not self.config_path.exists():
                logger.info("Config not found at %s", self.config_path)
                self._config_cache = {}
                self._last_mtime = 0
                return {}

            current_mtime = self.config_path.stat().st_mtime
            if not self._config_cache or force_reload or current_mtime > self._last_mtime:
                try:
                    raw = self._read_yaml()
                    validated = self._validate_and_migrate(raw)
                    self._config_cache = validated
                    self._last_mtime = current_mtime
                except ConfigError as ce:
                    logger.error("%s", ce, extra={"context": getattr(ce, "context", {})})
                    return {}
                except Exception as e:
                    logger.exception("Error loading config: %s", e)
                    return {}

            # deep copy via dump/load for immutability
            return yaml.safe_load(yaml.safe_dump(self._config_cache, sort_keys=False)) or {}

    def save_config(self, config: dict[str, Any]) -> bool:
        """
        Persistently save a configuration by deep-merging the provided values into the existing config, validating and writing the result to config/main.yaml.
        
        Parameters:
            config (dict[str, Any]): Partial or full configuration values to merge into the existing configuration.
        
        Returns:
            bool: `True` if the merged and validated configuration was written successfully, `False` otherwise. Invalid configurations are rejected and not written; writes are performed atomically and an existing file is moved to a `.yaml.bak` backup when present.
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

                # Atomic write with backup
                fd, tmp_path = tempfile.mkstemp(
                    prefix="main.yaml.", dir=str(self.config_path.parent)
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
                        except Exception:
                            pass
        except ConfigError as ce:
            logger.error(
                "Refusing to save invalid config: %s",
                ce,
                extra={"context": getattr(ce, "context", {})},
            )
            return False
        except Exception as e:
            logger.exception("Error saving config: %s", e)
            return False

    def get_env_info(self) -> dict[str, str]:
        """
        Collect environment-derived metadata from layered .env files and the process environment.
        
        Loads .env and .env.local (with .env.local overriding) and returns non-sensitive metadata.
        
        Returns:
            env_info (dict[str, str]): Mapping of metadata keys to values. Contains "model" derived from the
                LLM_MODEL environment key or the default "Pro/Flash" from DEFAULTS if unset.
        """
        env_path = self.project_root / ".env"
        local_path = self.project_root / ".env.local"
        parsed_env = self._load_env_file(env_path)
        parsed_env.update(self._load_env_file(local_path))

        def _get(key: str, default: str = "") -> str:
            return str(parsed_env.get(key) or os.environ.get(key, default))

        return {
            "model": _get("LLM_MODEL", DEFAULTS.get("llm", {}).get("model", "Pro/Flash")),
        }

    def validate_required_env(self, keys: list[str]) -> dict[str, list[str]]:
        """
        Check whether the specified environment variable names are present in the layered environment sources (.env, .env.local) or the process environment.
        
        Parameters:
            keys (list[str]): Environment variable names to validate.
        
        Returns:
            dict[str, list[str]]: A mapping with key "missing" containing the list of variable names that were not found.
        """
        env_path = self.project_root / ".env"
        local_path = self.project_root / ".env.local"
        parsed_env = self._load_env_file(env_path)
        parsed_env.update(self._load_env_file(local_path))
        missing = [k for k in keys if not (parsed_env.get(k) or os.environ.get(k))]
        if missing:
            logger.warning("Missing required env keys", extra={"missing": missing})
        return {"missing": missing}

    @classmethod
    def reset_for_tests(cls) -> None:
        """Reset singleton to allow re-initialization in tests with a different project_root."""
        with cls._lock:
            cls._instance = None