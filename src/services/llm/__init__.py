"""Unified LLM service exports for DeepTutor modules."""

# Also expose the providers for direct access if needed
from . import cloud_provider, local_provider
from .capabilities import (
    DEFAULT_CAPABILITIES,
    MODEL_OVERRIDES,
    PROVIDER_CAPABILITIES,
    get_capability,
    has_thinking_tags,
    requires_api_version,
    supports_response_format,
    supports_streaming,
    supports_tools,
    system_in_messages,
)
from .client import LLMClient, get_llm_client, reset_llm_client
from .config import (
    LLMConfig,
    get_llm_config,
    get_token_limit_kwargs,
    uses_max_completion_tokens,
)
from .exceptions import (
    LLMAPIError,
    LLMAuthenticationError,
    LLMConfigError,
    LLMError,
    LLMModelNotFoundError,
    LLMProviderError,
    LLMRateLimitError,
    LLMTimeoutError,
)
from .factory import (
    API_PROVIDER_PRESETS,
    DEFAULT_EXPONENTIAL_BACKOFF,
    DEFAULT_MAX_RETRIES,
    DEFAULT_RETRY_DELAY,
    LOCAL_PROVIDER_PRESETS,
    complete,
    fetch_models,
    get_provider_presets,
    stream,
)
from .utils import (
    build_auth_headers,
    build_chat_url,
    clean_thinking_tags,
    extract_response_content,
    is_local_llm_server,
    sanitize_url,
)

__all__ = [
    # Client (legacy, prefer factory functions)
    "LLMClient",
    "get_llm_client",
    "reset_llm_client",
    # Config
    "LLMConfig",
    "get_llm_config",
    "uses_max_completion_tokens",
    "get_token_limit_kwargs",
    # Capabilities
    "PROVIDER_CAPABILITIES",
    "MODEL_OVERRIDES",
    "DEFAULT_CAPABILITIES",
    "get_capability",
    "supports_response_format",
    "supports_streaming",
    "system_in_messages",
    "has_thinking_tags",
    "supports_tools",
    "requires_api_version",
    # Exceptions
    "LLMError",
    "LLMConfigError",
    "LLMProviderError",
    "LLMAPIError",
    "LLMTimeoutError",
    "LLMRateLimitError",
    "LLMAuthenticationError",
    "LLMModelNotFoundError",
    # Factory (main API)
    "complete",
    "stream",
    "fetch_models",
    "get_provider_presets",
    "API_PROVIDER_PRESETS",
    "LOCAL_PROVIDER_PRESETS",
    # Retry configuration
    "DEFAULT_MAX_RETRIES",
    "DEFAULT_RETRY_DELAY",
    "DEFAULT_EXPONENTIAL_BACKOFF",
    # Providers
    "cloud_provider",
    "local_provider",
    # Utils
    "sanitize_url",
    "is_local_llm_server",
    "build_chat_url",
    "build_auth_headers",
    "clean_thinking_tags",
    "extract_response_content",
]
