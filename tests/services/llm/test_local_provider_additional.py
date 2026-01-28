"""Tests for local provider behavior."""

from __future__ import annotations

import importlib
from types import TracebackType

from _pytest.monkeypatch import MonkeyPatch
import pytest

local_provider = importlib.import_module("src.services.llm.local_provider")


class _AsyncIterator:
    def __init__(self, items: list[bytes]) -> None:
        self._items = items
        self._index = 0

    def __aiter__(self):
        return self

    async def __anext__(self) -> bytes:
        if self._index >= len(self._items):
            raise StopAsyncIteration
        item = self._items[self._index]
        self._index += 1
        return item


class _FakeResponse:
    def __init__(self, status: int, json_data: object) -> None:
        self.status = status
        self._json_data = json_data
        self.content = _AsyncIterator([])

    async def __aenter__(self):
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        return None

    async def json(self) -> object:
        return self._json_data

    async def text(self) -> str:
        return ""


class _FakeStreamResponse(_FakeResponse):
    def __init__(self, status: int, lines: list[bytes]) -> None:
        super().__init__(status, {})
        self.content = _AsyncIterator(lines)


class _FakeSession:
    def __init__(self, response: _FakeResponse) -> None:
        self._response = response

    async def __aenter__(self):
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        return None

    def post(self, _url: str, **_kwargs: object) -> _FakeResponse:
        return self._response

    def get(self, _url: str, **_kwargs: object) -> _FakeResponse:
        return self._response


@pytest.mark.asyncio
async def test_local_complete(monkeypatch: MonkeyPatch) -> None:
    """Local complete should parse content from choices."""
    response = _FakeResponse(200, {"choices": [{"message": {"content": "ok"}}]})
    monkeypatch.setattr(
        local_provider.aiohttp,
        "ClientSession",
        lambda *args, **kwargs: _FakeSession(response),
    )

    result = await local_provider.complete(
        prompt="hello",
        model="model",
        base_url="http://localhost:1234/v1",
    )

    assert result == "ok"


@pytest.mark.asyncio
async def test_local_stream(monkeypatch: MonkeyPatch) -> None:
    """Local stream should yield chunks from SSE data lines."""
    lines = [
        b"data: {\"choices\": [{\"delta\": {\"content\": \"hi\"}}]}\n\n",
        b"data: [DONE]\n\n",
    ]
    response = _FakeStreamResponse(200, lines)
    monkeypatch.setattr(
        local_provider.aiohttp,
        "ClientSession",
        lambda *args, **kwargs: _FakeSession(response),
    )

    chunks = []
    async for chunk in local_provider.stream(
        prompt="hello",
        model="model",
        base_url="http://localhost:1234/v1",
    ):
        chunks.append(chunk)

    assert "".join(chunks) == "hi"


@pytest.mark.asyncio
async def test_local_fetch_models(monkeypatch: MonkeyPatch) -> None:
    """Model fetch should return models from the local endpoint response."""
    response = _FakeResponse(200, {"models": [{"name": "a"}, {"name": "b"}]})
    monkeypatch.setattr(
        local_provider.aiohttp,
        "ClientSession",
        lambda *args, **kwargs: _FakeSession(response),
    )

    models = await local_provider.fetch_models("http://localhost:11434/v1")

    assert models == ["a", "b"]


@pytest.mark.asyncio
async def test_local_fetch_models_openai_style(monkeypatch: MonkeyPatch) -> None:
    """OpenAI-style model payloads should be parsed correctly."""
    response = _FakeResponse(200, {"data": [{"id": "m1"}, {"id": "m2"}]})
    monkeypatch.setattr(
        local_provider.aiohttp,
        "ClientSession",
        lambda *args, **kwargs: _FakeSession(response),
    )

    models = await local_provider.fetch_models("http://localhost:1234/v1")

    assert models == ["m1", "m2"]


@pytest.mark.asyncio
async def test_local_stream_fallback(monkeypatch: MonkeyPatch) -> None:
    """Stream should fall back to complete on unexpected errors."""

    class _FailingSession:
        async def __aenter__(self):
            raise RuntimeError("boom")

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

    async def _fake_complete(**_kwargs: object) -> str:
        return "fallback"

    monkeypatch.setattr(
        local_provider.aiohttp,
        "ClientSession",
        lambda *args, **kwargs: _FailingSession(),
    )
    monkeypatch.setattr(local_provider, "complete", _fake_complete)

    chunks = []
    async for chunk in local_provider.stream(
        prompt="hello",
        model="model",
        base_url="http://localhost:1234/v1",
    ):
        chunks.append(chunk)

    assert chunks == ["fallback"]
