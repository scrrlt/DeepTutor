from typing import Any, Dict, Optional


class BaseError(Exception):
    def __init__(self, message: str, *, context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.context = context or {}


class ConfigError(BaseError):
    pass


class EnvError(BaseError):
    pass
