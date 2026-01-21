# -*- coding: utf-8 -*-
"""Tests for LLM error mapping helpers."""

import sys
import types

# Prevent optional provider imports from failing at collection time.
mod = types.ModuleType("src.services.search.providers")
mod.get_available_providers = lambda: []
mod.get_default_provider = lambda: "perplexity"
mod.get_provider = lambda name: types.SimpleNamespace(
    name=name,
    supports_answer=True,
    search=lambda query, **kwargs: types.SimpleNamespace(
        to_dict=lambda: {"answer": "", "citations": [], "search_results": []}
    ),
)
mod.get_providers_info = lambda: []
mod.list_providers = lambda: []
sys.modules.setdefault("src.services.search.providers", mod)

from src.services.llm.error_mapping import map_error
from src.services.llm.exceptions import ProviderContextWindowError


def test_context_window_error_mapping() -> None:
    """Ensure context length errors map to ProviderContextWindowError.

    Use a representative stub exception rather than a bare Exception so the
    test remains resilient to changes in classifier logic that rely on type
    attributes or additional fields.
    """

    class StubContextError(Exception):
        def __str__(self) -> str:  # pragma: no cover - trivial
            return "maximum context length exceeded"

    mapped = map_error(StubContextError("maximum context length exceeded"), provider="openai")
    assert isinstance(mapped, ProviderContextWindowError)
