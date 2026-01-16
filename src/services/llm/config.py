"""
LLM Configuration Management.
Uses Pydantic Settings for validation and type safety.
"""

from __future__ import annotations

from typing import Any

from pydantic import (
    AliasChoices,
    Field,
    SecretStr,
    ValidationError,
    computed_field,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

from .exceptions import LLMConfigError


PROVIDER_API_BASE_URLS: dict[str, str] = {
    "ollama": "http://localhost:11434/v1",
    "lm_studio": "http://localhost:1234/v1",
    "openai": "https://api.openai.com/v1",
    "anthropic": "https://api.anthropic.com/v1",
    "azure_openai": "",
}


class LLMConfig(BaseSettings):
    """
    Immutable configuration for LLM services.
    Reads from env vars with prefix LLM_ (e.g. LLM_MODEL, LLM_API_KEY).
    """

    model_config = SettingsConfigDict(
        env_prefix="LLM_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # Ignore undefined env vars
        case_sensitive=False,
        populate_by_name=True,
        frozen=True,
    )

    # Core fields
    # Default chosen to keep the factory usable out-of-the-box in dev/test
    # without forcing an environment variable.
    model: str = Field(
        "gpt-4o-mini",
        description="The model identifier (e.g., gpt-4o, gpt-4o-mini)",
    )
    # Defaults
    binding: str = Field(
        "openai",
        validation_alias=AliasChoices("binding", "provider_name"),
        description="Provider binding (openai, anthropic, ollama)",
    )
    base_url: str | None = Field(
        None, description="Base API URL for the provider"
    )
    api_key: SecretStr | None = Field(
        None, description="API Key (optional for local models)"
    )
    api_version: str | None = Field(
        None, description="API Version (Azure/OpenAI specific)"
    )

    # Ops controls
    timeout: float = Field(120.0, gt=0.0)
    max_concurrency: int = Field(20, ge=1)
    requests_per_minute: int = Field(600, ge=1)

    # Tuning
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(4096, gt=0)

    @computed_field
    def effective_url(self) -> str:
        """
        Return explicit base_url or infer from binding.

        Returns:
            Effective API URL string.
        """
        if self.base_url:
            return self.base_url

        url = PROVIDER_API_BASE_URLS.get(self.binding)
        if url is not None:
            if url:
                return url

        if self.binding.startswith(("http://", "https://")):
            return self.binding

        raise LLMConfigError(
            f"Unknown binding '{self.binding}' requires explicit base_url"
        )

    @property
    def provider_name(self) -> str:
        """
        Alias for binding to maintain compatibility.

        Returns:
            Provider binding name.
        """
        return self.binding

    def get_api_key(self) -> str | None:
        """Safely access the raw API key value."""
        if self.api_key is None:
            return None
        return self.api_key.get_secret_value()


# Global cache for the settings instance
_settings: LLMConfig | None = None


def get_llm_config() -> LLMConfig:
    """
    Singleton access to LLM Settings.

    Returns:
        The valid LLM configuration instance.

    Raises:
        LLMConfigError: If configuration is invalid.
    """
    global _settings
    if _settings is None:
        try:
            _settings = LLMConfig()
        except ValidationError as e:
            raise LLMConfigError(f"Configuration error: {e}") from e
    return _settings


def reload_config() -> LLMConfig:
    """
    Force reload configuration from environment.

    Returns:
        New LLM configuration instance.

    Raises:
        LLMConfigError: If configuration is invalid.
    """
    global _settings
    try:
        _settings = LLMConfig()
    except ValidationError as e:
        raise LLMConfigError(f"Configuration error: {e}") from e
    return _settings


def clear_llm_config_cache() -> None:
    """Clear the cached singleton without validating environment."""
    global _settings
    _settings = None


__all__ = [
    "LLMConfig",
    "get_llm_config",
    "reload_config",
    "clear_llm_config_cache",
]
