# -*- coding: utf-8 -*-
"""
Error Mapping - Map provider-specific errors to unified exceptions.
"""

from __future__ import annotations

from dataclasses import dataclass
import functools
import importlib
import importlib.util
from types import ModuleType
from typing import Callable, Optional, Type, cast

from .exceptions import (
    LLMAPIError,
    LLMAuthenticationError,
    LLMError,
    LLMRateLimitError,
    ProviderContextWindowError,
)

ErrorClassifier = Callable[[Exception], bool]


@dataclass(frozen=True)
class MappingRule:
    """Classifier/factory pair for mapping external errors.

    Args:
        classifier: Predicate that recognizes an error.
        factory: Factory that produces an LLMError.

    Returns:
        None.

    Raises:
        None.
    """

    classifier: ErrorClassifier
    factory: Callable[[Exception, Optional[str]], LLMError]


def _instance_of(*types: Type[BaseException]) -> ErrorClassifier:
    return lambda exc: isinstance(exc, types)


def _message_contains(*needles: str) -> ErrorClassifier:
    def _classifier(exc: Exception) -> bool:
        msg = str(exc).lower()
        return any(needle in msg for needle in needles)

    return _classifier


_CORE_RULES: list[MappingRule] = [
    MappingRule(
        classifier=_message_contains("rate limit", "429", "quota"),
        factory=lambda exc, provider: LLMRateLimitError(str(exc), provider=provider),
    ),
    MappingRule(
        classifier=_message_contains("context length", "maximum context"),
        factory=lambda exc, provider: ProviderContextWindowError(str(exc), provider=provider),
    ),
]


@functools.lru_cache(maxsize=1)
def get_mapping_rules() -> list[MappingRule]:
    """Build error mapping rules lazily based on available SDKs.

    Returns:
        List of mapping rules.

    Raises:
        None.
    """
    rules = list(_CORE_RULES)

    if importlib.util.find_spec("openai"):
        openai_module = cast(ModuleType, importlib.import_module("openai"))
        rules[:0] = [
            MappingRule(
                classifier=_instance_of(openai_module.AuthenticationError),
                factory=lambda exc, provider: LLMAuthenticationError(
                    str(exc),
                    provider=provider,
                ),
            ),
            MappingRule(
                classifier=_instance_of(openai_module.RateLimitError),
                factory=lambda exc, provider: LLMRateLimitError(str(exc), provider=provider),
            ),
        ]

    if importlib.util.find_spec("anthropic"):
        anthropic_module = cast(ModuleType, importlib.import_module("anthropic"))
        rules.append(
            MappingRule(
                classifier=_instance_of(anthropic_module.RateLimitError),
                factory=lambda exc, provider: LLMRateLimitError(str(exc), provider=provider),
            )
        )

    return rules


def map_error(exc: Exception, provider: Optional[str] = None) -> LLMError:
    """Map provider-specific errors to unified internal exceptions.

    Args:
        exc: Exception to map.
        provider: Optional provider identifier.

    Returns:
        Normalized LLMError instance.

    Raises:
        None.
    """
    status_code = getattr(exc, "status_code", None)
    if status_code == 401:
        return LLMAuthenticationError(str(exc), provider=provider)
    if status_code == 429:
        return LLMRateLimitError(str(exc), provider=provider)

    for rule in get_mapping_rules():
        if rule.classifier(exc):
            return rule.factory(exc, provider)

    return LLMAPIError(str(exc), status_code=status_code, provider=provider)
