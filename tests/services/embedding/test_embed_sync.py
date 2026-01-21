# -*- coding: utf-8 -*-
import pytest
import asyncio

from src.services.embedding.client import EmbeddingClient


@pytest.mark.asyncio
async def test_embed_sync_deadlock_detection():
    # Create a dummy instance without running __init__
    client = object.__new__(EmbeddingClient)
    # Simulate initialization loop being the current loop
    client._init_loop = asyncio.get_running_loop()

    with pytest.raises(RuntimeError) as excinfo:
        client.embed_sync(["test"])

    assert "Deadlock Risk" in str(excinfo.value)
