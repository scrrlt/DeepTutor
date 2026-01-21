import asyncio
import types
import sys

import pytest

# Prevent optional provider imports during test collection
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

from fastapi import WebSocketDisconnect

from src.api.routers import chat


class DummyWebSocket:
    def __init__(self):
        self._inbox = [
            {
                "message": "hello",
                "session_id": None,
                "kb_name": "",
                "enable_rag": False,
                "enable_web_search": False,
            }
        ]
        self.sent = []
        self._accepted = False

    async def accept(self):
        self._accepted = True

    async def receive_json(self):
        if not self._inbox:
            raise WebSocketDisconnect()
        return self._inbox.pop(0)

    async def send_json(self, data):
        self.sent.append(data)


@pytest.mark.asyncio
async def test_stream_non_iterable_logs_and_notifies_client(monkeypatch):
    # Replace ChatAgent with a dummy that returns a non-async iterable for process
    class DummyAgent:
        def __init__(self, *args, **kwargs):
            pass

        async def refresh_config(self):
            return None

        async def process(self, *args, **kwargs):
            # Deliberately return a dict (non-async iterable) to trigger the warning path
            return {"response": "not a stream"}

    monkeypatch.setattr(chat, "ChatAgent", DummyAgent)

    ws = DummyWebSocket()

    # Run websocket_chat until WebSocketDisconnect occurs
    await chat.websocket_chat(ws)

    # Assert that client was notified of the non-iterable stream (error message present)
    assert any(
        msg.get("type") == "error" and "streaming" in (msg.get("message") or "") for msg in ws.sent
    )
