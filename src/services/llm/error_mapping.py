# -*- coding: utf-8 -*-
"""
Error Mapping - Map provider-specific errors to unified exceptions.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
import logging
from types import ModuleType
from typing import Callable, List, Optional, Type, cast

# Import unified exceptions from exceptions.py
from .exceptions import (
    LLMAPIError,
    LLMAuthenticationError,
    LLMError,
    LLMRateLimitError,
    LLMServiceUnavailableError,
    LLMTimeoutError,
    ProviderContextWindowError,
)

try:
    import openai  # type: ignore

    _HAS_OPENAI = True
except ImportError:  # pragma: no cover
    openai = None  # type: ignore
    _HAS_OPENAI = False


logger = logging.getLogger(__name__)


ErrorClassifier = Callable[[BaseException], bool]


class MappingRule:
    """Mapping rule pairing classifier and factory."""

    def __init__(
        self,
        classifier: ErrorClassifier,
        factory: Callable[[BaseException, str | None], LLMError],
    ) -> None:
        self.classifier = classifier
        self.factory = factory


def _instance_of(*types: type[BaseException]) -> ErrorClassifier:
    return lambda exc: isinstance(exc, types)


def _message_contains(*needles: str) -> ErrorClassifier:
    def _classifier(exc: BaseException) -> bool:
        msg = str(exc).lower()
        return any(needle in msg for needle in needles)

    return _classifier


def _iter_sdk_rules() -> Iterable[MappingRule]:
    """Yield SDK-specific rules first (most reliable)."""
    if _HAS_OPENAI and openai is not None:
        yield MappingRule(
            classifier=_instance_of(getattr(openai, "AuthenticationError")),
            factory=lambda exc, provider: LLMAuthenticationError(
                str(exc),
                provider=provider,
                details={"source": "openai"},
            ),
        )
        yield MappingRule(
            classifier=_instance_of(getattr(openai, "RateLimitError")),
            factory=lambda exc, provider: LLMRateLimitError(
                str(exc),
                provider=provider,
                details={"source": "openai"},
            ),
        )

    try:
        import anthropic  # type: ignore

        yield MappingRule(
            classifier=_instance_of(getattr(anthropic, "RateLimitError")),
            factory=lambda exc, provider: LLMRateLimitError(
                str(exc),
                provider=provider,
                details={"source": "anthropic"},
            ),
        )
        yield MappingRule(
            classifier=_instance_of(getattr(anthropic, "APITimeoutError")),
            factory=lambda exc, provider: LLMTimeoutError(
                str(exc),
                provider=provider,
            ),
        )
    except ImportError:  # pragma: no cover
        pass

    try:
        import httpx  # type: ignore

        yield MappingRule(
            classifier=_instance_of(
                getattr(httpx, "ReadTimeout"),
                getattr(httpx, "ConnectTimeout"),
                getattr(httpx, "TimeoutException"),
            ),
            factory=lambda exc, provider: LLMTimeoutError(
                str(exc),
                provider=provider,
            ),
        )
    except ImportError:  # pragma: no cover
        pass


_HEURISTIC_RULES: tuple[MappingRule, ...] = (
    MappingRule(
        classifier=_message_contains("context length", "maximum context"),
        factory=lambda exc, provider: ProviderContextWindowError(
            str(exc), provider=provider
        ),
    ),
]

if _HAS_OPENAI and openai is not None:
    openai_module = cast(ModuleType, openai)
    _GLOBAL_RULES[:0] = [
        MappingRule(
            classifier=_instance_of(openai_module.AuthenticationError),
            factory=lambda exc, provider: LLMAuthenticationError(str(exc), provider=provider),
        ),
        MappingRule(
            classifier=_instance_of(openai_module.RateLimitError),
            factory=lambda exc, provider: LLMRateLimitError(str(exc), provider=provider),
        ),
    ]

# Attempt to load Anthropic and Google rules if SDKs are present
try:
    import anthropic

    _GLOBAL_RULES.append(
        MappingRule(
            classifier=_instance_of(anthropic.RateLimitError),
            factory=lambda exc, provider: LLMRateLimitError(str(exc), provider=provider),
        )
    )
except ImportError:
    pass


def map_error(exc: Exception, provider: Optional[str] = None) -> LLMError:
    """Map provider-specific errors to unified internal exceptions."""
    # Heuristic check for status codes before rules
    status_code = getattr(exc, "status_code", None)
    request_id = getattr(exc, "request_id", None)

    if status_code == 401:
        return LLMAuthenticationError(
            str(exc),
            provider=provider,
            request_id=str(request_id) if request_id else None,
        )
    if status_code == 429:
        return LLMRateLimitError(
            str(exc),
            provider=provider,
            request_id=str(request_id) if request_id else None,
        )
    if status_code == 503:
        return LLMServiceUnavailableError(
            str(exc),
            provider=provider,
            request_id=str(request_id) if request_id else None,
        )

    for rule in _iter_sdk_rules():
        try:
            if rule.classifier(exc):
                return rule.factory(exc, provider)
        except Exception as rule_exc:
            logger.debug("Error mapping rule failed: %s", rule_exc)

    for rule in _HEURISTIC_RULES:
        if rule.classifier(exc):
            return rule.factory(exc, provider)

    return LLMAPIError(
        str(exc),
        status_code=status_code,
        provider=provider,
        request_id=str(request_id) if request_id else None,
        details={"original_type": type(exc).__name__},
    )
