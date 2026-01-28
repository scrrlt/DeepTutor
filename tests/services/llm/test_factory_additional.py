"""Tests for factory routing behavior."""

import sys
import types

import pytest

from src.services.llm import factory
from src.services.llm.exceptions import LLMAPIError, LLMAuthenticationError


@pytest.mark.asyncio
async def test_factory_complete_routes_local(monkeypatch) -> None:
    """Factory should route to local provider for local URLs."""

    async def _fake_complete(**_kwargs: object) -> str:
        return "local"

    monkeypatch.setattr(factory.local_provider, "complete", _fake_complete)

    result = await factory.complete(
        prompt="hello",
        model="model",
        api_key="",
        base_url="http://localhost:1234",
    )

    assert result == "local"


@pytest.mark.asyncio
async def test_factory_stream_routes_cloud(monkeypatch) -> None:
    """Factory should route to cloud provider for remote URLs."""

    async def _fake_stream(**_kwargs: object):
        yield "chunk"

    fake_module = types.ModuleType("src.services.llm.cloud_provider")
    fake_module.stream = _fake_stream
    monkeypatch.setitem(sys.modules, "src.services.llm.cloud_provider", fake_module)
    llm_package = sys.modules["src.services.llm"]
    monkeypatch.setattr(llm_package, "cloud_provider", fake_module, raising=False)

    chunks = []
    async for chunk in factory.stream(
        prompt="hello",
        model="model",
        api_key="",
        base_url="https://api.openai.com",
    ):
        chunks.append(chunk)

    assert chunks == ["chunk"]


def test_factory_retriable_error_checks() -> None:
    """Retriable error helper should honor status codes and auth errors."""
    assert factory._is_retriable_error(LLMAuthenticationError("auth")) is False
    assert factory._is_retriable_error(LLMAPIError("server", status_code=503)) is True
    assert factory._is_retriable_error(LLMAPIError("client", status_code=400)) is False


@pytest.mark.asyncio
async def test_factory_fetch_models_routes_local(monkeypatch) -> None:
    """Factory fetch_models should route to local provider for local URLs."""

    async def _fake_fetch_models(_base_url: str, _api_key: str | None = None) -> list[str]:
        return ["local"]

    monkeypatch.setattr(factory.local_provider, "fetch_models", _fake_fetch_models)

    models = await factory.fetch_models("local", "http://localhost:11434/v1")

    assert models == ["local"]


@pytest.mark.asyncio
async def test_factory_fetch_models_routes_cloud(monkeypatch) -> None:
    """Factory fetch_models should route to cloud provider for remote URLs."""

    async def _fake_fetch_models(
        _base_url: str,
        _api_key: str | None = None,
        _binding: str = "openai",
    ) -> list[str]:
        return ["cloud"]

    fake_module = types.ModuleType("src.services.llm.cloud_provider")
    fake_module.fetch_models = _fake_fetch_models
    monkeypatch.setitem(sys.modules, "src.services.llm.cloud_provider", fake_module)
    llm_package = sys.modules["src.services.llm"]
    monkeypatch.setattr(llm_package, "cloud_provider", fake_module, raising=False)

    models = await factory.fetch_models("openai", "https://api.openai.com")

    assert models == ["cloud"]
