"""Error Mapping - Map provider-specific errors to unified exceptions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple, Type

import openai

from .exceptions import (
    LLMBaseError,
    ProviderAuthenticationError,
    ProviderContextWindowError,
    ProviderQuotaExceededError,
)


ErrorClassifier = Callable[[Exception], bool]


@dataclass(frozen=True)
class MappingRule:
    classifier: ErrorClassifier
    factory: Callable[[Exception], LLMBaseError]


def _instance_of(*types: Type[BaseException]) -> ErrorClassifier:
    return lambda exc: isinstance(exc, types)


def _message_contains(*needles: str) -> ErrorClassifier:
    def _classifier(exc: Exception) -> bool:
        msg = str(exc).lower()
        return any(needle in msg for needle in needles)

    return _classifier


_GLOBAL_RULES: List[MappingRule] = [
    MappingRule(
        classifier=_instance_of(openai.AuthenticationError),
        factory=lambda exc: ProviderAuthenticationError(str(exc)),
    ),
    MappingRule(
        classifier=_instance_of(openai.RateLimitError),
        factory=lambda exc: ProviderQuotaExceededError(str(exc)),
    ),
    MappingRule(
        classifier=_instance_of(openai.InvalidRequestError),
        factory=lambda exc: ProviderContextWindowError(str(exc)),
    ),
    MappingRule(
        classifier=_message_contains("rate limit", "429", "quota"),
        factory=lambda exc: ProviderQuotaExceededError(str(exc)),
    ),
    MappingRule(
        classifier=_message_contains("context length", "maximum context"),
        factory=lambda exc: ProviderContextWindowError(str(exc)),
    ),
]

# Optional SDK-specific mappings (Anthropic, Google Generative AI)
try:  # Anthropic SDK
    import anthropic

    _GLOBAL_RULES.extend(
        [
            MappingRule(
                classifier=_instance_of(anthropic.RateLimitError),
                factory=lambda exc: ProviderQuotaExceededError(str(exc)),
            ),
            MappingRule(
                classifier=_instance_of(anthropic.APIStatusError),
                factory=lambda exc: ProviderQuotaExceededError(str(exc))
                if getattr(exc, "status_code", None) in {429, 503}
                else LLMBaseError(str(exc)),
            ),
        ]
    )
except ImportError:
    pass

try:  # Google Generative AI / google.api_core exceptions
    from google.api_core import exceptions as google_exceptions

    _GLOBAL_RULES.extend(
        [
            MappingRule(
                classifier=_instance_of(
                    google_exceptions.ResourceExhausted,
                    google_exceptions.TooManyRequests,
                ),
                factory=lambda exc: ProviderQuotaExceededError(str(exc)),
            ),
            MappingRule(
                classifier=_instance_of(
                    google_exceptions.InvalidArgument,
                    google_exceptions.FailedPrecondition,
                ),
                factory=lambda exc: ProviderContextWindowError(str(exc)),
            ),
        ]
    )
except ImportError:
    pass


_PROVIDER_RULES: Dict[str, Tuple[MappingRule, ...]] = {}


def register_mapping(provider: str, *rules: MappingRule) -> None:
    provider_key = provider.lower()
    existing = _PROVIDER_RULES.get(provider_key, tuple())
    _PROVIDER_RULES[provider_key] = existing + rules


def map_error(exc: Exception, provider: str | None = None) -> LLMBaseError:
    """Map provider-specific errors to unified exceptions."""
    rules = list(_GLOBAL_RULES)
    if provider:
        rules.extend(_PROVIDER_RULES.get(provider.lower(), tuple()))

    for rule in rules:
        if rule.classifier(exc):
            return rule.factory(exc)

    return LLMBaseError(str(exc))
