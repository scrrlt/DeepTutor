from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.services.embedding.adapters.base import (
    EmbeddingRequest,
    EmbeddingResponse,
)
from src.services.embedding.client import EmbeddingClient
from src.services.embedding.config import (
    EmbeddingConfig,
    _strip_value,
    _to_bool,
    _to_int,
)

# --- Config Tests ---


def test_config_helpers():
    assert _strip_value(None) is None
    assert _strip_value(' "test" ') == "test"

    assert _to_int("10", 5) == 10
    assert _to_int("invalid", 5) == 5
    assert _to_int(None, 5) == 5

    assert _to_bool("true", False) is True
    assert _to_bool("YES", False) is True
    assert _to_bool("1", False) is True
    assert _to_bool("other", True) is False
    assert _to_bool(None, True) is True


def test_embedding_config_initialization():
    config = EmbeddingConfig(model="test-model", api_key="sk-key", dim=1536)
    assert config.model == "test-model"
    assert config.dim == 1536


# --- Client Tests ---


@pytest.fixture
def mock_embedding_config():
    return EmbeddingConfig(model="test-model", api_key="sk-test", binding="openai", dim=1536)


@pytest.fixture
def mock_adapter():
    adapter = MagicMock()
    adapter.embed = AsyncMock(
        return_value=EmbeddingResponse(
            embeddings=[[0.1, 0.2]],
            model="test-model",
            dimensions=2,
            usage={"prompt_tokens": 10, "total_tokens": 10},
        )
    )
    return adapter


@pytest.fixture
def client(mock_embedding_config, mock_adapter):
    with patch("src.services.embedding.client.get_embedding_provider_manager") as mock_mgr_get:
        mock_mgr = MagicMock()
        mock_mgr.get_adapter.return_value = mock_adapter
        mock_mgr_get.return_value = mock_mgr

        client = EmbeddingClient(config=mock_embedding_config)
        yield client


@pytest.mark.asyncio
async def test_embed_success(client, mock_adapter):
    texts = ["hello"]
    embeddings = await client.embed(texts)

    assert embeddings == [[0.1, 0.2]]
    mock_adapter.embed.assert_called_once()
    args = mock_adapter.embed.call_args[0][0]
    assert isinstance(args, EmbeddingRequest)
    assert args.texts == texts


def test_embed_sync_success(client, mock_adapter):
    texts = ["hello"]
    embeddings = client.embed_sync(texts)
    assert embeddings == [[0.1, 0.2]]


@pytest.mark.asyncio
async def test_embed_validation_error(client, mock_adapter):
    # Mock invalid response
    mock_adapter.embed.return_value = EmbeddingResponse(
        embeddings=[],  # Empty
        model="test",
        dimensions=2,
        usage={},
    )
    with pytest.raises(ValueError, match="Embeddings response is empty or invalid."):
        await client.embed(["test"])

    mock_adapter.embed.return_value = EmbeddingResponse(
        embeddings=[[]],  # Empty inner list
        model="test",
        dimensions=2,
        usage={},
    )
    # The validation logic in client checks for empty lists
    with pytest.raises(ValueError, match="Embeddings response is empty or invalid."):
        await client.embed(["test"])


@pytest.mark.asyncio
async def test_embed_manager_error(mock_embedding_config):
    with patch("src.services.embedding.client.get_embedding_provider_manager") as mock_mgr_get:
        mock_mgr = MagicMock()
        mock_mgr.get_adapter.side_effect = ValueError("Unknown provider")
        mock_mgr_get.return_value = mock_mgr

        with pytest.raises(ValueError, match="Unknown provider"):
            EmbeddingClient(config=mock_embedding_config)
