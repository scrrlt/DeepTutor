"""Provider-aware client helpers for LLM and embedding API calls."""

from __future__ import annotations

import asyncio
import re
from concurrent.futures import ProcessPoolExecutor
from contextvars import ContextVar
from pathlib import Path
from typing import Literal

from openai import AsyncOpenAI
from pydantic import BaseModel
from yaml import safe_load

try:
    from deeptutor.exceptions import ConfigurationValidationError
except ImportError:

    class ConfigurationValidationError(Exception):
        """Fallback for missing ConfigurationValidationError."""

# PEP 695 type aliases for provider identifiers
type ProviderName = Literal[
    "openai",
    "gemini",
    "openrouter",
    "groq",
    "together",
    "deepseek",
    "xai",
    "mistral",
    "ollama",
    "custom",
]

type ClientKind = Literal["llm", "embedding"]


def get_settings() -> object:
    """Fallback settings accessor that merges defaults with environment overrides."""
    try:
        from deeptutor.config import get_settings as _get_settings

        return _get_settings()
    except ImportError:
        from deeptutor.config.settings import settings

        return settings


class CompletionPayload(BaseModel):
    """Strict Pydantic contract for LLM completion parameters."""

    model: str
    messages: list[dict[str, str]]
    temperature: float = 0.78
    max_tokens: int = 1200
    logit_bias: dict[str, int] | None = None


_PROVIDER_BASE_URLS: dict[ProviderName, str] = {
    "openai": "https://api.openai.com/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai",
    "openrouter": "https://openrouter.ai/api/v1",
    "groq": "https://api.groq.com/openai/v1",
    "together": "https://api.together.xyz/v1",
    "deepseek": "https://api.deepseek.com/v1",
    "xai": "https://api.x.ai/v1",
    "mistral": "https://api.mistral.ai/v1",
    "ollama": "http://localhost:11434/v1",
    "custom": "",
}

_DEFAULT_LLM_MODELS: dict[ProviderName, str] = {
    "openai": "gpt-4o-mini",
    "gemini": "gemini-2.0-flash",
    "openrouter": "openai/gpt-4o-mini",
    "groq": "llama-3.3-70b-versatile",
    "together": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "deepseek": "deepseek-chat",
    "xai": "grok-2-latest",
    "mistral": "mistral-small-latest",
    "ollama": "llama3.1:8b",
    "custom": "gpt-4o-mini",
}

_DEFAULT_EMBEDDING_MODELS: dict[ProviderName, str] = {
    "openai": "text-embedding-3-small",
    "gemini": "gemini-embedding-001",
    "openrouter": "text-embedding-3-small",
    "groq": "text-embedding-3-small",
    "together": "togethercomputer/m2-bert-80M-32k-retrieval",
    "deepseek": "text-embedding-3-small",
    "xai": "text-embedding-3-small",
    "mistral": "mistral-embed",
    "ollama": "nomic-embed-text",
    "custom": "text-embedding-3-small",
}


class ProviderClientRegistry:
    """Runtime registry for provider-aware API clients."""

    def __init__(self) -> None:
        self.llm_client: AsyncOpenAI | None = None
        self.embedding_client: AsyncOpenAI | None = None


_registry_ctx: ContextVar[ProviderClientRegistry] = ContextVar(
    "provider_client_registry", default=ProviderClientRegistry()
)

_logit_bias_cache: dict[str, int] | None = None
_logit_bias_lock = asyncio.Lock()
_process_pool: ProcessPoolExecutor | None = None


def _get_process_pool() -> ProcessPoolExecutor:
    """Lazy initialization of the process pool for CPU-bound tasks."""
    global _process_pool
    if _process_pool is None:
        _process_pool = ProcessPoolExecutor(max_workers=1)
    return _process_pool


def get_registry() -> ProviderClientRegistry:
    """Return the request-isolated provider client registry."""
    return _registry_ctx.get()


def set_provider_client_registry(registry: ProviderClientRegistry) -> None:
    """Inject a provider client registry into the current context (useful for tests)."""
    _registry_ctx.set(registry)


def _resolve_provider(kind: ClientKind) -> ProviderName:
    settings = get_settings()
    return settings.llm_provider if kind == "llm" else settings.embedding_provider


def _resolved_provider(
    kind: ClientKind,
    provider: ProviderName | None = None,
) -> ProviderName:
    return provider or _resolve_provider(kind)


def _resolve_base_url(
    kind: ClientKind,
    provider: ProviderName | None = None,
) -> str:
    settings = get_settings()
    resolved_provider = _resolved_provider(kind, provider)
    explicit = (
        settings.llm_api_base_url if kind == "llm" else settings.embedding_api_base_url
    )
    if explicit:
        return explicit

    base_url = _PROVIDER_BASE_URLS[resolved_provider]
    if resolved_provider == "custom" and not base_url:
        raise ConfigurationValidationError(
            f"{kind.upper()}_API_BASE_URL is required when provider is 'custom'"
        )
    return base_url


def _provider_key(provider: ProviderName) -> str | None:
    settings = get_settings()
    if provider == "openai":
        return settings.openai_api_key
    if provider == "gemini":
        return settings.gemini_api_key
    if provider == "openrouter":
        return settings.openrouter_api_key
    if provider == "groq":
        return settings.groq_api_key
    if provider == "together":
        return settings.together_api_key
    if provider == "deepseek":
        return settings.deepseek_api_key
    if provider == "xai":
        return settings.xai_api_key
    if provider == "mistral":
        return settings.mistral_api_key
    if provider == "ollama":
        return "ollama"
    return settings.openai_api_key


def has_provider_api_key(provider: ProviderName) -> bool:
    """Return True when an API key is configured for the provider."""
    return _provider_key(provider) is not None


def _resolve_api_key(
    kind: ClientKind,
    provider: ProviderName | None = None,
) -> str:
    settings = get_settings()
    resolved_provider = _resolved_provider(kind, provider)
    explicit = settings.llm_api_key if kind == "llm" else settings.embedding_api_key
    if explicit and provider is None:
        return explicit

    fallback = _provider_key(resolved_provider)
    if fallback:
        return fallback

    if resolved_provider == "custom":
        raise ValueError(
            f"{kind.upper()}_API_KEY is required when provider is 'custom'"
        )

    env_hint = f"{resolved_provider.upper()}_API_KEY"
    raise ValueError(
        "Missing API key for provider "
        f"'{resolved_provider}'. Set {env_hint} or {kind.upper()}_API_KEY."
    )


def _make_client(
    kind: ClientKind,
    provider: ProviderName | None = None,
) -> AsyncOpenAI:
    settings = get_settings()
    timeout_seconds = float(getattr(settings, "provider_request_timeout_seconds", 10.0))
    return AsyncOpenAI(
        api_key=_resolve_api_key(kind, provider),
        base_url=_resolve_base_url(kind, provider),
        timeout=timeout_seconds,
    )


def get_llm_client() -> AsyncOpenAI:
    """Return request-isolated cached provider-aware LLM client."""
    registry = get_registry()
    if registry.llm_client is None:
        registry.llm_client = _make_client("llm")
    return registry.llm_client


def get_embedding_client() -> AsyncOpenAI:
    """Return request-isolated cached provider-aware embedding client."""
    registry = get_registry()
    if registry.embedding_client is None:
        registry.embedding_client = _make_client("embedding")
    return registry.embedding_client


def create_llm_client(provider: ProviderName) -> AsyncOpenAI:
    """Create a non-cached LLM client for an explicit provider."""
    return _make_client("llm", provider)


def create_embedding_client(provider: ProviderName) -> AsyncOpenAI:
    """Create a non-cached embedding client for an explicit provider."""
    return _make_client("embedding", provider)


def reset_provider_clients() -> None:
    """Clear cached provider clients for the current context."""
    global _logit_bias_cache
    registry = get_registry()
    registry.llm_client = None
    registry.embedding_client = None
    _logit_bias_cache = None


def resolve_llm_model() -> str:
    """Resolve model name with provider-aware defaults."""
    settings = get_settings()
    provider = settings.llm_provider
    if provider != "openai" and settings.llm_model == _DEFAULT_LLM_MODELS["openai"]:
        return _DEFAULT_LLM_MODELS[provider]
    return settings.llm_model


def resolve_llm_model_for_provider(provider: ProviderName) -> str:
    """Resolve the chat model for an explicit provider override."""
    settings = get_settings()
    if provider == settings.llm_provider:
        return resolve_llm_model()
    if provider == settings.llm_fallback_provider and settings.llm_fallback_model:
        return settings.llm_fallback_model
    return _DEFAULT_LLM_MODELS[provider]


def get_llm_fallback_provider() -> ProviderName | None:
    """Return the configured fallback provider when it is usable."""
    settings = get_settings()
    provider = settings.llm_fallback_provider
    if provider is None or provider == settings.llm_provider:
        return None
    if not has_provider_api_key(provider):
        return None
    return provider


def get_llm_provider_sequence() -> list[ProviderName]:
    """Return primary→fallback provider order for resilient completion calls."""
    settings = get_settings()
    sequence: list[ProviderName] = [settings.llm_provider]
    fallback = get_llm_fallback_provider()
    if fallback is not None and fallback not in sequence:
        sequence.append(fallback)
    return sequence


def get_llm_model_sequence() -> list[tuple[ProviderName, str]]:
    """Return ordered (provider, model) pairs for completion failover."""
    return [
        (provider, resolve_llm_model_for_provider(provider))
        for provider in get_llm_provider_sequence()
    ]


def resolve_embedding_model() -> str:
    """Resolve embedding model with provider-aware defaults."""
    settings = get_settings()
    provider = settings.embedding_provider
    if (
        provider != "openai"
        and settings.embedding_model == _DEFAULT_EMBEDDING_MODELS["openai"]
    ):
        return _DEFAULT_EMBEDDING_MODELS[provider]
    return settings.embedding_model


def get_embedding_dimensions() -> int | None:
    """Dynamically resolve embedding dimensions from the model catalog or settings."""
    try:
        from deeptutor.services.config.model_catalog import get_model_catalog_service

        catalog_svc = get_model_catalog_service()
        catalog = catalog_svc.load()
        active_model = catalog_svc.get_active_model(catalog, "embedding")
        if active_model:
            dim = active_model.get("dimension")
            if isinstance(dim, (int, float)):
                return int(dim)
            if isinstance(dim, str) and dim.isdigit():
                return int(dim)
    except Exception:
        pass

    settings = get_settings()
    # Support explicit setting override if present
    return getattr(settings, "embedding_dimensions", None)


def should_send_embedding_dimensions() -> bool:
    """Return True when dimensions is safe for the selected provider."""
    provider = _resolve_provider("embedding")
    return provider in {"openai", "gemini", "openrouter", "custom"}


def _normalise_pattern_to_term(pattern: str) -> str | None:
    value = pattern.replace("\\b", "").replace("\\", "")
    value = value.strip().lower()
    if not value:
        return None
    if re.search(r"[\[\]{}()|+*?^$]", value):
        return None
    value = re.sub(r"\s+", " ", value)
    return value


def _load_blocked_terms() -> list[str]:
    """Extract high-severity vocabulary and phrase terms from ai_patterns config."""
    cfg_path = Path(__file__).resolve().parent.parent / "config" / "ai_patterns.yml"
    if not cfg_path.exists():
        return []

    data_obj = safe_load(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(data_obj, dict):
        return []
    extracted_terms: list[str] = []
    for section_name in ("vocabulary", "phrases", "ngram_phrases"):
        section_obj = data_obj.get(section_name)
        if not isinstance(section_obj, list):
            continue

        for item in section_obj:
            if not isinstance(item, dict):
                continue
            if item.get("severity") != "high":
                continue
            pattern_obj = item.get("pattern")
            if not isinstance(pattern_obj, str):
                continue

            normalised = _normalise_pattern_to_term(pattern_obj)
            if normalised is not None:
                extracted_terms.append(normalised)

    deduped: list[str] = []
    seen: set[str] = set()
    for term in extracted_terms:
        if term in seen:
            continue
        seen.add(term)
        deduped.append(term)
    return deduped


def _token_ids_for_blocklist(terms: list[str]) -> list[int]:
    """Encode blocked terms into token IDs with tiktoken when available."""
    try:
        import tiktoken
    except ImportError:
        return []

    encoding = tiktoken.get_encoding("cl100k_base")
    token_ids: list[int] = []
    for term in terms:
        encoded = encoding.encode(term)
        token_ids.extend(encoded)

    unique_ids: list[int] = []
    seen: set[int] = set()
    for token_id in token_ids:
        if token_id in seen:
            continue
        seen.add(token_id)
        unique_ids.append(token_id)
    return unique_ids


def build_logit_bias_map() -> dict[str, int]:
    """Build a suppression map for high-severity filler vocabulary tokens (sync fallback)."""
    global _logit_bias_cache
    if _logit_bias_cache is not None:
        return _logit_bias_cache

    blocked_terms = _load_blocked_terms()
    token_ids = _token_ids_for_blocklist(blocked_terms)
    _logit_bias_cache = {str(token_id): -100 for token_id in token_ids}
    return _logit_bias_cache


async def get_logit_bias_map() -> dict[str, int]:
    """Asynchronous, non-blocking retrieval of the logit-bias map."""
    global _logit_bias_cache
    if _logit_bias_cache is not None:
        return _logit_bias_cache

    async with _logit_bias_lock:
        if _logit_bias_cache is not None:
            return _logit_bias_cache

        loop = asyncio.get_running_loop()
        executor = _get_process_pool()
        blocked_terms = await loop.run_in_executor(executor, _load_blocked_terms)
        token_ids = await loop.run_in_executor(executor, _token_ids_for_blocklist, blocked_terms)
        _logit_bias_cache = {str(token_id): -100 for token_id in token_ids}
        return _logit_bias_cache


def prepare_completion_payload(
    *,
    model: str,
    messages: list[dict[str, str]],
    temperature: float = 0.78,
    max_tokens: int = 1200,
) -> CompletionPayload:
    """Construct completion payload with active logit-bias suppression."""
    logit_bias = build_logit_bias_map()
    return CompletionPayload(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        logit_bias=logit_bias or None,
    )
