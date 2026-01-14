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


def _strip_value(val: Optional[str]) -> Optional[str]:
    """Strip whitespace from value if not None."""
    return val.strip() if val else None


def _get_llm_config_from_env() -> LLMConfig:
    """Get LLM configuration from environment variables."""
    binding = _strip_value(os.getenv("LLM_BINDING", "openai"))
    model = _strip_value(os.getenv("LLM_MODEL"))
    api_key = _strip_value(os.getenv("LLM_API_KEY"))
    base_url = _strip_value(os.getenv("LLM_HOST"))
    api_version = _strip_value(os.getenv("LLM_API_VERSION"))

    return LLMConfig(
        provider_name=binding,
        api_key=api_key,
        base_url=base_url,
        model_name=model,
        api_version=api_version,
    )

async def get_llm_config_async() -> LLMConfig:
    """
    Async version of get_llm_config for non-blocking configuration loading.

    Load LLM configuration from environment variables or provider manager.

    The behavior depends on the LLM_MODE environment variable:
    - hybrid (default): Use active provider if available, else env config
    - api: Only use API providers (active API provider or env config)
    - local: Only use local providers (active local provider or env config)

    Priority:
    1. Active provider from llm_providers.json (if mode compatible)
    2. Environment variables (.env)

    Returns:
        LLMConfig: Configuration dataclass

    Raises:
        ValueError: If required configuration is missing
    """
    mode = _get_llm_mode_str()

    # 1. Try to get active provider from provider manager
    try:
        from .provider import provider_manager

        active_provider = await provider_manager.get_active_provider_async()

        if active_provider:
            provider_is_local = getattr(active_provider, "provider_type", "local") == "local"

            # Check mode compatibility
            use_provider = False
            if mode == LLM_MODE_HYBRID:
                use_provider = True
            elif mode == LLM_MODE_API and not provider_is_local:
                use_provider = True
            elif mode == LLM_MODE_LOCAL and provider_is_local:
                use_provider = True

            if use_provider:
                return LLMConfig(
                    binding=active_provider.binding,
                    model=active_provider.model,
                    api_key=active_provider.api_key,
                    base_url=active_provider.base_url,
                    api_version=getattr(active_provider, "api_version", None),
                    provider_type=getattr(active_provider, "provider_type", "local"),
                )
    except Exception as e:
        logger.warning(f"Failed to load active provider: {e}")

    # 2. Fallback to environment variables (no async needed since these are env vars)
    return _get_llm_config_from_env()


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
