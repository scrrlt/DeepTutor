import contextlib

import pytest
from unittest.mock import AsyncMock, patch

from src.services.rag.pipelines.raganything import RAGAnythingPipeline


@pytest.fixture
def pipeline(tmp_path):
    return RAGAnythingPipeline(kb_base_dir=str(tmp_path))


@pytest.mark.asyncio
async def test_initialize_calls_engine(pipeline):
    rag = AsyncMock()
    rag._ensure_lightrag_initialized = AsyncMock()
    rag.process_document_complete = AsyncMock()

    with (
        patch.object(pipeline, "_get_rag_instance", return_value=rag),
        patch(
            "src.services.rag.pipelines.raganything.LightRAGLogContext",
            return_value=contextlib.nullcontext(),
        ),
    ):
        ok = await pipeline.initialize(
            kb_name="kb",
            file_paths=["/tmp/a.pdf", "/tmp/b.pdf"],
            extract_numbered_items=False,
        )

    assert ok is True
    rag._ensure_lightrag_initialized.assert_awaited_once()
    assert rag.process_document_complete.await_count == 2