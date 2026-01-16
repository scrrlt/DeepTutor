"""
LLM Configuration Management.
Uses Pydantic Settings for validation and type safety.
"""

from typing import Any

from pydantic import (
    Field,
    ValidationError,
    computed_field,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

from .exceptions import LLMConfigError


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
        frozen=True,
    )

    # Required Fields
    model: str = Field(..., description="The model identifier (e.g., gpt-4o)")
    # Defaults
    binding: str = Field(
        "openai",
        description="Provider binding (openai, anthropic, ollama)",
    )
    base_url: str | None = Field(
        None, description="Base API URL for the provider"
    )
    api_key: str | None = Field(
        None, description="API Key (optional for local models)"
    )
    api_version: str | None = Field(
        None, description="API Version (Azure/OpenAI specific)"
    )

    # Tuning
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(4096, gt=0)

    @model_validator(mode="before")
    @classmethod
    def alias_provider_name(cls, data: Any) -> Any:
        """
        Allow provider_name as an alias for binding in constructor.

        Args:
            data: Raw input data for model validation.

        Returns:
            Validated data with binding set if provider_name was present.
        """
        if isinstance(data, dict):
            if "provider_name" in data and "binding" not in data:
                data["binding"] = data["provider_name"]
        return data

    @computed_field
    def effective_url(self) -> str:
        """
        Return explicit base_url or infer from binding.

        Returns:
            Effective API URL string.
        """
        if self.base_url:
            return self.base_url

        # Fallback presets (Centralized here or in a separate presets module)
        presets = {
            "ollama": "http://localhost:11434/v1",
            "lm_studio": "http://localhost:1234/v1",
            "openai": "https://api.openai.com/v1",
            "anthropic": "https://api.anthropic.com/v1",
        }
        url = presets.get(self.binding)
        if url is None:
            raise LLMConfigError(
                f"Unknown binding '{self.binding}' requires explicit base_url"
            )
        return url

    @property
    def is_reasoning_model(self) -> bool:
        """
        Detect if model requires reasoning-specific parameters (o1, o3).

        Returns:
            True if the model is a reasoning model, False otherwise.
        """
        return self.model.startswith(("o1", "o3"))

    @property
    def token_param_name(self) -> str:
        """
        Resolve the correct parameter name for token limits.

        Returns:
            Parameter name ('max_completion_tokens' or 'max_tokens').
        """
        # 1. Reasoning models use max_completion_tokens
        if self.is_reasoning_model:
            return "max_completion_tokens"

        # 2. GPT-4o and newer often prefer max_completion_tokens in newer SDKs,
        #    but max_tokens is usually aliased. Strictly forcing it for 4o:
        if self.model.lower().startswith("gpt-4o"):
            return "max_completion_tokens"

        return "max_tokens"

    @property
    def provider_name(self) -> str:
        """
        Alias for binding to maintain compatibility.

        Returns:
            Provider binding name.
        """
        return self.binding


# Global cache for the settings instance
_settings: LLMConfig | None = None


def get_llm_config() -> LLMConfig:
    """
    Singleton access to LLM Settings.

    Returns:
        The valid LLM configuration instance.

    Raises:
        LLMConfigError: If required env vars (LLM_MODEL) are missing.
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


# Compatibility shims
def uses_max_completion_tokens(model: str) -> bool:
    """
    Check if the model uses max_completion_tokens via temporary config.

    Note: This creates a temporary config instance which is slightly inefficient
    but maintains API compatibility.

    Args:
        model: Model identifier.

    Returns:
        True if max_completion_tokens is required.
    """
    return model.startswith(("o1", "o3")) or model.lower().startswith("gpt-4o")


def get_token_limit_kwargs(model: str, max_tokens: int) -> dict[str, int]:
    """Get the appropriate token limit parameter for the model."""
    if uses_max_completion_tokens(model):
        return {"max_completion_tokens": max_tokens}
    return {"max_tokens": max_tokens}


__all__ = [
    "LLMConfig",
    "get_llm_config",
    "reload_config",
    "uses_max_completion_tokens",
    "get_token_limit_kwargs",
]
