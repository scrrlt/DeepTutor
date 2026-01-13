"""
Base exception classes for consistent error handling across the application.
Provides a standardized way to distinguish between bugs, recoverable errors,
and configuration issues.
"""

from typing import Any, Dict, Optional


class DeepTutorError(Exception):
    """Base class for all application errors in DeepTutor."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} (details: {self.details})"
        return self.message


class ConfigurationError(DeepTutorError):
    """Raised when there's a configuration-related error."""

    pass


class ValidationError(DeepTutorError):
    """Raised when input validation fails."""

    pass


class ServiceError(DeepTutorError):
    """Base class for service layer errors."""

    pass


class LLMServiceError(ServiceError):
    """Base class for LLM service-related errors."""

    pass


class LLMContextError(LLMServiceError):
    """Raised when prompt exceeds model context window."""

    pass
from typing import Any, Dict, Optional

class BaseError(Exception):
    def __init__(self, message: str, *, context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.context = context or {}

class ConfigError(BaseError):
    pass

class EnvError(BaseError):
    pass

