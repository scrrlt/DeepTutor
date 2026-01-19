# -*- coding: utf-8 -*-
"""Tests for LLM factory routing behavior."""

import pytest

from src.services.llm import cloud_provider, factory, local_provider


@pytest.mark.asyncio
async def test_complete_routes_to_local(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_local(**_kwargs: object) -> str:
        return "local"

    async def fake_cloud(**_kwargs: object) -> str:
        raise AssertionError("Cloud provider should not be called")

    monkeypatch.setattr(local_provider, "complete", fake_local)
    monkeypatch.setattr(cloud_provider, "complete", fake_cloud)

    result = await factory.complete(
        prompt="hi",
        model="local-model",
        base_url="http://localhost:1234/v1",
    )

    assert result == "local"


@pytest.mark.asyncio
async def test_complete_routes_to_cloud(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_local(**_kwargs: object) -> str:
        raise AssertionError("Local provider should not be called")

    async def fake_cloud(**_kwargs: object) -> str:
        return "cloud"

    monkeypatch.setattr(local_provider, "complete", fake_local)
    monkeypatch.setattr(cloud_provider, "complete", fake_cloud)

    result = await factory.complete(
        prompt="hi",
        model="cloud-model",
        base_url="https://api.openai.com/v1",
    )

    assert result == "cloud"


@pytest.mark.asyncio
async def test_stream_routes_to_local(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_local_stream(**_kwargs: object):
        yield "local-1"
        yield "local-2"

    async def fake_cloud_stream(**_kwargs: object):
        if False:
            yield "unused"
        raise AssertionError("Cloud provider should not be called")

    monkeypatch.setattr(local_provider, "stream", fake_local_stream)
    monkeypatch.setattr(cloud_provider, "stream", fake_cloud_stream)

    chunks = [
        chunk
        async for chunk in factory.stream(
            prompt="hi",
            model="local-model",
            base_url="http://localhost:1234/v1",
        )
    ]

    assert chunks == ["local-1", "local-2"]


@pytest.mark.asyncio
async def test_stream_routes_to_cloud(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_local_stream(**_kwargs: object):
        if False:
            yield "unused"
        raise AssertionError("Local provider should not be called")

    async def fake_cloud_stream(**_kwargs: object):
        yield "cloud-1"
        yield "cloud-2"

    monkeypatch.setattr(local_provider, "stream", fake_local_stream)
    monkeypatch.setattr(cloud_provider, "stream", fake_cloud_stream)

    chunks = [
        chunk
        async for chunk in factory.stream(
            prompt="hi",
            model="cloud-model",
            base_url="https://api.openai.com/v1",
        )
    ]

    assert chunks == ["cloud-1", "cloud-2"]
