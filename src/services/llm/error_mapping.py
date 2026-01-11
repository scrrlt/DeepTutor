"""
Error Mapping - Map provider-specific errors to unified exceptions.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Callable, List, Optional, Type

import openai

from .exceptions import (
    LLMAPIError,
    LLMAuthenticationError,
    LLMError,
    LLMRateLimitError,
    ProviderContextWindowError,
    ProviderQuotaExceededError,
)

logger = logging.getLogger(__name__)

ErrorClassifier = Callable[[Exception], bool]


@dataclass(frozen=True)
class MappingRule:
    classifier: ErrorClassifier
    factory: Callable[[Exception], LLMError]


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
        factory=lambda exc: LLMAuthenticationError(str(exc)),
    ),
    MappingRule(
        classifier=_instance_of(openai.RateLimitError),
        factory=lambda exc: ProviderQuotaExceededError(str(exc)),
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

# Attempt to load Anthropic and Google rules if SDKs are present
try:
    import anthropic

    _GLOBAL_RULES.append(
        MappingRule(
            classifier=_instance_of(anthropic.RateLimitError),
            factory=lambda exc: ProviderQuotaExceededError(str(exc)),
        )
    )
except ImportError:
    pass


def map_error(exc: Exception, provider: Optional[str] = None) -> LLMError:
    """Map provider-specific errors to unified internal exceptions."""
    # Heuristic check for status codes before rules
    status_code = getattr(exc, "status_code", None)
    if status_code == 401:
        return LLMAuthenticationError(str(exc), provider=provider)
    if status_code == 429:
        return LLMRateLimitError(str(exc), provider=provider)

    for rule in _GLOBAL_RULES:
        if rule.classifier(exc):
            return rule.factory(exc)

    return LLMAPIError(str(exc), status_code=status_code, provider=provider)
