from dataclasses import dataclass, field
import os
from typing import Optional, Dict


@dataclass
class LLMConfig:
    """
    Unified Configuration Object for all LLM Providers.
    Automatically fetches ENV vars based on provider name.
    """
    provider_name: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model_name: Optional[str] = None
    api_version: Optional[str] = None
    env_prefix: str = field(init=False)
    pricing: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        # Determine ENV prefix (e.g. "OPENAI", "AZURE", "ANTHROPIC")
        self.env_prefix = self.provider_name.upper()

        # 1. Load API Key (if not passed explicitly)
        if not self.api_key:
            self.api_key = os.getenv(f"{self.env_prefix}_API_KEY")

        # 2. Load Base URL
        if not self.base_url:
            self.base_url = os.getenv(f"{self.env_prefix}_BASE_URL") or os.getenv(
                f"{self.env_prefix}_ENDPOINT"
                )
        if not self.base_url:
            self.base_url = os.getenv("LLM_BASE_URL")

        # 3. Load Model Name
        if not self.model_name:
            self.model_name = os.getenv(f"{self.env_prefix}_MODEL")
        if not self.model_name:
            self.model_name = os.getenv("LLM_MODEL")

        # 4. Load API Version (Common for Azure)
        if not self.api_version:
            self.api_version = os.getenv(f"{self.env_prefix}_API_VERSION")

        # 5. Load Pricing (optional, for cost calculation)
        if not self.pricing:
            input_price = os.getenv(f"{self.env_prefix}_PRICE_PER_INPUT_TOKEN")
            output_price = os.getenv(f"{self.env_prefix}_PRICE_PER_OUTPUT_TOKEN")
            if input_price and output_price:
                self.pricing = {
                    "input": float(input_price),
                    "output": float(output_price)
                }

    @property
    def is_valid(self) -> bool:
        """Basic validation check"""
        # Local providers like Ollama might not need API keys
        if self.provider_name == "ollama":
            return True
        return bool(self.api_key)

    @property
    def model(self) -> Optional[str]:
        return self.model_name

    @model.setter
    def model(self, value: Optional[str]) -> None:
        self.model_name = value


_llm_config: Optional[LLMConfig] = None


def _get_llm_config_from_env() -> LLMConfig:
    """Get LLM configuration from environment variables."""
    binding = _strip_value(os.getenv("LLM_BINDING", "openai"))
    model = _strip_value(os.getenv("LLM_MODEL"))
    api_key = _strip_value(os.getenv("LLM_API_KEY"))
    base_url = _strip_value(os.getenv("LLM_HOST"))
    api_version = _strip_value(os.getenv("LLM_API_VERSION"))

    # Validate required configuration
    if not model:
        raise LLMConfigError(
            "LLM_MODEL not set, please configure it in .env file or activate a provider"
        )

    # Determine provider type from base_url
    from .utils import is_local_llm_server

    provider_type: Literal["api", "local"] = "local" if is_local_llm_server(base_url) else "api"

    # Check if API key is required (not required for local providers)
    requires_key = (
        provider_type == "api" or os.getenv("LLM_API_KEY_REQUIRED", "false").lower() == "true"
    )

    if requires_key and not api_key:
        raise LLMConfigError(
            "LLM_API_KEY not set, please configure it in .env file or activate a provider"
        )
    if not base_url:
        raise LLMConfigError(
            "LLM_HOST not set, please configure it in .env file or activate a provider"
        )

    return LLMConfig(
        binding=binding,
        model=model,
        api_key=api_key or "",
        base_url=base_url,
        api_version=api_version,
        provider_type=provider_type,
    )


def get_llm_config() -> LLMConfig:
    """Return a singleton LLMConfig based on environment variables."""
    global _llm_config
    if _llm_config is None:
        provider = os.getenv("LLM_PROVIDER", "openai")
        _llm_config = LLMConfig(provider)
    return _llm_config


def uses_max_completion_tokens(provider: str | None = None) -> bool:
    """Return whether this provider expects max_completion_tokens instead of max_tokens."""
    # Conservative default: most OpenAI-compatible providers accept max_tokens.
    # This function exists mainly for API compatibility with existing call sites.
    _ = provider
    return False


def get_token_limit_kwargs(
    model: str | int | None = None,
    max_tokens: int | None = None,
    binding: str | None = None,
) -> Dict[str, int]:
    """Build token-limit kwargs for provider calls.

    Compatibility:
    - Newer call sites use: get_token_limit_kwargs(model, max_tokens)
    - Older call sites may use: get_token_limit_kwargs(max_tokens)
    """
    # Back-compat: if called as get_token_limit_kwargs(200)
    if isinstance(model, int) and max_tokens is None:
        max_tokens = model
        model = None

    if max_tokens is None:
        return {}

    if binding is None:
        try:
            cfg = get_llm_config()
            binding = getattr(cfg, "provider_name", None) or "openai"
        except Exception:
            binding = "openai"

    try:
        from .capabilities import get_capability

        use_max_completion = bool(
            get_capability(
                binding,
                "newer_models_use_max_completion_tokens",
                model=str(model) if model is not None else None,
                default=False,
            )
        )
    except Exception:
        use_max_completion = False

    if use_max_completion:
        return {"max_completion_tokens": int(max_tokens)}
    return {"max_tokens": int(max_tokens)}
