# -*- coding: utf-8 -*-
import pytest
import importlib
from types import SimpleNamespace

# Load module under package name to ensure relative imports resolve
openai_mod = importlib.import_module("src.services.embedding.adapters.openai_compatible")
EmbeddingRequest = openai_mod.EmbeddingRequest
OpenAIAdapter = openai_mod.OpenAICompatibleEmbeddingAdapter


class FakeResponse:
    def __init__(self, status_code=200, json_data=None, text=""):
        self.status_code = status_code
        self._json = json_data if json_data is not None else {}
        self.text = text

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception("HTTP error")

    def json(self):
        return self._json


class FakeAsyncClient:
    def __init__(self, response: FakeResponse):
        self._response = response

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, *args, **kwargs):
        return self._response


@pytest.mark.asyncio
async def test_openai_compatible_missing_data_raises(monkeypatch):
    adapter = OpenAIAdapter({"base_url": "http://localhost", "model": "text-embedding-3-small"})

    fake_resp = FakeResponse(status_code=200, json_data={})
    monkeypatch.setattr(
        "src.services.embedding.adapters.openai_compatible.httpx.AsyncClient",
        lambda *args, **kwargs: FakeAsyncClient(fake_resp),
    )

    req = EmbeddingRequest(texts=["test"], model="text-embedding-3-small")

    with pytest.raises(ValueError):
        await adapter.embed(req)
