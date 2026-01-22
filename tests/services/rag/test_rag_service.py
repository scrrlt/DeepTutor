#!/usr/bin/env python

"""
Tests for the RAGService.
"""

import json
from unittest.mock import AsyncMock, MagicMock, mock_open, patch

import pytest

from src.services.rag.service import RAGService


@pytest.fixture
def rag_service():
    """
    Provides a RAGService instance with mocked dependencies.
    """
    with patch("src.services.rag.service.get_logger"):
        service = RAGService(kb_base_dir="/tmp/test_kb", provider="test_provider")
        yield service


@patch("src.services.rag.factory.get_pipeline")
def test_rag_service_initialization(mock_get_pipeline, rag_service: RAGService):
    """
    Tests that the RAGService can be initialized correctly.
    """
    assert rag_service.kb_base_dir == "/tmp/test_kb"
    assert rag_service.provider == "test_provider"
    assert rag_service._pipeline is None

    # Test that get_pipeline is called when the pipeline is accessed
    rag_service._get_pipeline()
    mock_get_pipeline.assert_called_once_with("test_provider", kb_base_dir="/tmp/test_kb")


@pytest.mark.asyncio
@patch("src.services.rag.factory.get_pipeline")
async def test_rag_service_initialize(mock_get_pipeline, rag_service: RAGService):
    """
    Tests that the initialize method calls the pipeline's initialize method.
    """
    mock_pipeline = AsyncMock()
    mock_get_pipeline.return_value = mock_pipeline

    await rag_service.initialize("test_kb", ["file1.txt", "file2.txt"])

    mock_pipeline.initialize.assert_called_once_with(
        kb_name="test_kb", file_paths=["file1.txt", "file2.txt"]
    )


@pytest.mark.asyncio
@patch("src.services.rag.factory.get_pipeline")
async def test_rag_service_search(mock_get_pipeline, rag_service: RAGService):
    """
    Tests that the search method calls the pipeline's search method.
    """
    mock_pipeline = AsyncMock()
    mock_get_pipeline.return_value = mock_pipeline

    with patch.object(rag_service, "_get_provider_for_kb", return_value="test_provider"):
        await rag_service.search("test_query", "test_kb", mode="hybrid")

        mock_pipeline.search.assert_called_once_with(
            query="test_query", kb_name="test_kb", mode="hybrid"
        )


@pytest.mark.asyncio
@patch("src.services.rag.factory.get_pipeline")
async def test_rag_service_delete_with_pipeline_method(mock_get_pipeline, rag_service: RAGService):
    """
    Tests that the delete method calls the pipeline's delete method.
    """
    mock_pipeline = AsyncMock()
    mock_pipeline.delete = AsyncMock()
    mock_get_pipeline.return_value = mock_pipeline

    await rag_service.delete("test_kb")

    mock_pipeline.delete.assert_called_once_with(kb_name="test_kb")


@pytest.mark.asyncio
@patch("src.services.rag.factory.get_pipeline")
@patch("shutil.rmtree")
@patch("pathlib.Path.exists")
async def test_rag_service_delete_manual(
    mock_path_exists, mock_rmtree, mock_get_pipeline, rag_service: RAGService
):
    """
    Tests that the delete method deletes the directory manually if the pipeline has no delete method.
    """
    mock_pipeline = MagicMock()
    del mock_pipeline.delete  # Ensure the pipeline has no delete method
    mock_get_pipeline.return_value = mock_pipeline
    mock_path_exists.return_value = True

    await rag_service.delete("test_kb")

    mock_rmtree.assert_called_once_with(rag_service.kb_base_dir + "/test_kb")


@patch("builtins.open", new_callable=mock_open)
@patch("pathlib.Path.exists")
def test_get_provider_for_kb(mock_exists, mock_file, rag_service: RAGService):
    """
    Tests that the _get_provider_for_kb method correctly retrieves the provider from the knowledge base metadata.
    """
    mock_exists.return_value = True

    metadata = {"rag_provider": "metadata_provider"}
    mock_file.return_value.read.return_value = json.dumps(metadata)

    provider = rag_service._get_provider_for_kb("test_kb")
    assert provider == "metadata_provider"

    # Test fallback to instance provider
    mock_file.return_value.read.return_value = "{}"
    provider = rag_service._get_provider_for_kb("test_kb")
    assert provider == "test_provider"
