# -*- coding: utf-8 -*-
import pytest
import asyncio
from unittest.mock import MagicMock

from src.services.embedding.client import EmbeddingClient


@pytest.mark.asyncio
async def test_embed_sync_deadlock_detection():
    # Create a mock instance that only has the attributes needed for deadlock detection
    client = MagicMock(spec=EmbeddingClient)

    # Simulate initialization loop being the current loop
    client._init_loop = asyncio.get_running_loop()

    # Bind the real embed_sync method to our mock instance
    client.embed_sync = EmbeddingClient.embed_sync.__get__(client, EmbeddingClient)

    with pytest.raises(RuntimeError) as excinfo:
        client.embed_sync(["test"])

    assert "Deadlock Risk" in str(excinfo.value)
