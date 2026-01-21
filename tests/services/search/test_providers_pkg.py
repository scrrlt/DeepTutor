# -*- coding: utf-8 -*-
import pytest

from src.services.search import providers


def test_registry_has_perplexity():
    assert "perplexity" in providers.get_available_providers()


def test_get_provider_returns_instance(monkeypatch):
    p = providers.get_provider("perplexity")
    assert p.name == "perplexity"

    # Provide a fake client to avoid network calls in unit tests
    class FakeChoices:
        def __init__(self):
            self.message = type("Msg", (), {"content": "Answer"})
            self.finish_reason = "stop"

    class FakeCompletion:
        def __init__(self):
            self.choices = [FakeChoices()]
            self.model = "sonar"
            self.usage = None
            self.search_results = []
            self.citations = []

    class FakeChat:
        def __init__(self):
            self.completions = type("C", (), {"create": lambda *a, **k: FakeCompletion()})()

    fake_client = type("CClient", (), {"chat": FakeChat()})()
    p._client = fake_client

    resp = p.search("hello")
    assert resp.provider == "perplexity"

