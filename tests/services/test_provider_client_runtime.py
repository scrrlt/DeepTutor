"""Tests for provider-client runtime registry and reset behaviour."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from deeptutor.services.writing.provider_client import (
    ProviderClientRegistry,
    build_logit_bias_map,
    get_embedding_client,
    get_llm_client,
    get_llm_fallback_provider,
    get_llm_model_sequence,
    get_llm_provider_sequence,
    has_provider_api_key,
    reset_provider_clients,
    resolve_embedding_model,
    resolve_llm_model_for_provider,
    set_provider_client_registry,
    should_send_embedding_dimensions,
)


def test_provider_client_registry_caches_clients() -> None:
    registry = ProviderClientRegistry()
    set_provider_client_registry(registry)

    llm_client = MagicMock(name="llm")
    embedding_client = MagicMock(name="embedding")

    with patch(
        "deeptutor.services.writing.provider_client._make_client",
        side_effect=[llm_client, embedding_client],
    ) as make_client:
        assert get_llm_client() is llm_client
        assert get_llm_client() is llm_client
        assert get_embedding_client() is embedding_client
        assert get_embedding_client() is embedding_client

    assert make_client.call_count == 2


def test_reset_provider_clients_recreates_clients() -> None:
    registry = ProviderClientRegistry()
    set_provider_client_registry(registry)

    first_client = MagicMock(name="first")
    second_client = MagicMock(name="second")

    with patch(
        "deeptutor.services.writing.provider_client._make_client",
        side_effect=[first_client, second_client],
    ):
        assert get_llm_client() is first_client
        reset_provider_clients()
        assert get_llm_client() is second_client


def test_resolve_embedding_model_uses_gemini_default() -> None:
    settings = SimpleNamespace(
        embedding_provider="gemini",
        embedding_model="text-embedding-3-small",
    )

    with patch("deeptutor.services.writing.provider_client.get_settings", return_value=settings):
        assert resolve_embedding_model() == "gemini-embedding-001"


def test_should_send_embedding_dimensions_for_gemini() -> None:
    settings = SimpleNamespace(embedding_provider="gemini", llm_provider="gemini")

    with patch("deeptutor.services.writing.provider_client.get_settings", return_value=settings):
        assert should_send_embedding_dimensions() is True


def test_resolve_llm_model_for_fallback_provider_override() -> None:
    settings = SimpleNamespace(
        llm_provider="gemini",
        llm_model="gemini-2.0-flash",
        llm_fallback_provider="openai",
        llm_fallback_model="gpt-4o-mini",
    )

    with patch("deeptutor.services.writing.provider_client.get_settings", return_value=settings):
        assert resolve_llm_model_for_provider("openai") == "gpt-4o-mini"


def test_get_llm_fallback_provider_requires_key() -> None:
    settings = SimpleNamespace(llm_provider="gemini", llm_fallback_provider="openai")

    with (
        patch("deeptutor.services.writing.provider_client.get_settings", return_value=settings),
        patch(
            "deeptutor.services.writing.provider_client.has_provider_api_key",
            return_value=True,
        ),
    ):
        assert get_llm_fallback_provider() == "openai"


def test_has_provider_api_key_for_gemini() -> None:
    settings = SimpleNamespace(
        openai_api_key=None,
        gemini_api_key="present",
        openrouter_api_key=None,
        groq_api_key=None,
        together_api_key=None,
        deepseek_api_key=None,
        xai_api_key=None,
        mistral_api_key=None,
    )

    with patch("deeptutor.services.writing.provider_client.get_settings", return_value=settings):
        assert has_provider_api_key("gemini") is True


def test_build_logit_bias_map_includes_multi_token_terms() -> None:
    with (
        patch(
            "deeptutor.services.writing.provider_client._load_blocked_terms",
            return_value=["it is important to note", "delve"],
        ),
        patch(
            "deeptutor.services.writing.provider_client._token_ids_for_blocklist",
            return_value=[101, 202, 303],
        ),
    ):
        payload = build_logit_bias_map()

    assert payload == {"101": -100, "202": -100, "303": -100}


def test_get_llm_provider_sequence_returns_primary_then_fallback() -> None:
    settings = SimpleNamespace(llm_provider="gemini", llm_fallback_provider="openai")

    with (
        patch("deeptutor.services.writing.provider_client.get_settings", return_value=settings),
        patch(
            "deeptutor.services.writing.provider_client.has_provider_api_key",
            return_value=True,
        ),
    ):
        sequence = get_llm_provider_sequence()

    assert sequence == ["gemini", "openai"]


def test_get_llm_model_sequence_uses_provider_specific_models() -> None:
    with (
        patch(
            "deeptutor.services.writing.provider_client.get_llm_provider_sequence",
            return_value=["gemini", "openai"],
        ),
        patch(
            "deeptutor.services.writing.provider_client.resolve_llm_model_for_provider",
            side_effect=["gemini-2.0-flash", "gpt-4o-mini"],
        ),
    ):
        sequence = get_llm_model_sequence()

    assert sequence == [("gemini", "gemini-2.0-flash"), ("openai", "gpt-4o-mini")]


def test_reset_provider_clients_clears_logit_bias_cache() -> None:
    registry = ProviderClientRegistry()
    set_provider_client_registry(registry)
    build_logit_bias_map.cache_clear()

    with (
        patch(
            "deeptutor.services.writing.provider_client._load_blocked_terms",
            return_value=["delve"],
        ) as load_terms,
        patch(
            "deeptutor.services.writing.provider_client._token_ids_for_blocklist",
            return_value=[101],
        ),
    ):
        first = build_logit_bias_map()
        second = build_logit_bias_map()
        assert first == second
        assert load_terms.call_count == 1

        reset_provider_clients()
        third = build_logit_bias_map()
        assert third == {"101": -100}
        assert load_terms.call_count == 2
