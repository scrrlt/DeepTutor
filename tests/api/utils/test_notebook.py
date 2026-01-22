"""Tests for refactored notebook management system.

Verifies separation of concerns, dependency injection, and async safety.
"""

from pathlib import Path
import tempfile

import pytest

from src.api.utils.notebook_manager import NotebookManager
from src.api.utils.notebook_models import RecordType
from src.api.utils.notebook_storage import NotebookStorage


@pytest.fixture
def temp_dir() -> Path:
    """Create temporary directory for test notebooks."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def notebook_manager(temp_dir: Path) -> NotebookManager:
    """Create notebook manager with temporary storage."""
    return NotebookManager(str(temp_dir))


class TestNotebookStorage:
    """Test storage layer isolation and error handling."""

    @pytest.mark.asyncio
    async def test_ensure_index_exists(self, temp_dir: Path) -> None:
        """Verify index file creation."""
        storage = NotebookStorage(temp_dir)
        assert not storage.index_file.exists()

        await storage.ensure_index_exists()
        assert storage.index_file.exists()

    @pytest.mark.asyncio
    async def test_save_and_load_notebook(self, temp_dir: Path) -> None:
        """Verify notebook persistence."""
        storage = NotebookStorage(temp_dir)

        notebook = {
            "id": "test123",
            "name": "Test",
            "records": [],
            "created_at": 123.45,
            "updated_at": 123.45,
        }

        await storage.save_notebook(notebook)
        loaded = await storage.load_notebook("test123")

        assert loaded is not None
        assert loaded["id"] == notebook["id"]
        assert loaded["name"] == notebook["name"]

    @pytest.mark.asyncio
    async def test_delete_notebook(self, temp_dir: Path) -> None:
        """Verify notebook deletion."""
        storage = NotebookStorage(temp_dir)

        notebook = {"id": "test123", "name": "Test", "records": []}
        await storage.save_notebook(notebook)

        deleted = await storage.delete_notebook("test123")
        assert deleted is True

        loaded = await storage.load_notebook("test123")
        assert loaded is None


class TestNotebookManager:
    """Test high-level manager API and backward compatibility."""

    @pytest.mark.asyncio
    async def test_create_notebook(self, notebook_manager: NotebookManager) -> None:
        """Verify notebook creation through manager."""
        await notebook_manager.ensure_initialized()

        result = await notebook_manager.create_notebook(
            name="My Notebook",
            description="Test notebook",
            color="#FF0000",
        )

        assert result["name"] == "My Notebook"
        assert result["description"] == "Test notebook"
        assert result["color"] == "#FF0000"
        assert "id" in result
        assert len(result["id"]) == 8  # UUID_LENGTH

    @pytest.mark.asyncio
    async def test_list_notebooks(self, notebook_manager: NotebookManager) -> None:
        """Verify listing notebooks."""
        await notebook_manager.ensure_initialized()

        # Create multiple notebooks
        await notebook_manager.create_notebook("Notebook 1")
        await notebook_manager.create_notebook("Notebook 2")

        notebooks = await notebook_manager.list_notebooks()

        assert len(notebooks) == 2
        assert all("id" in nb and "name" in nb for nb in notebooks)

    @pytest.mark.asyncio
    async def test_add_record(self, notebook_manager: NotebookManager) -> None:
        """Verify record addition."""
        await notebook_manager.ensure_initialized()

        nb = await notebook_manager.create_notebook("Test NB")
        notebook_id = nb["id"]

        result = await notebook_manager.add_record(
            notebook_ids=[notebook_id],
            record_type=RecordType.SOLVE,
            title="Solution",
            user_query="Solve this",
            output="Solution output",
            kb_name="test_kb",
        )

        assert result["record"]["type"] == RecordType.SOLVE
        assert result["added_to_notebooks"] == [notebook_id]

        # Verify record persisted
        updated_nb = await notebook_manager.get_notebook(notebook_id)
        assert len(updated_nb["records"]) == 1
        assert updated_nb["records"][0]["title"] == "Solution"

    @pytest.mark.asyncio
    async def test_remove_record(self, notebook_manager: NotebookManager) -> None:
        """Verify record removal."""
        await notebook_manager.ensure_initialized()

        nb = await notebook_manager.create_notebook("Test NB")
        nb_id = nb["id"]

        result = await notebook_manager.add_record(
            notebook_ids=[nb_id],
            record_type=RecordType.QUESTION,
            title="Question",
            user_query="Ask",
            output="Answer",
        )

        record_id = result["record"]["id"]

        # Remove the record
        removed = await notebook_manager.remove_record(nb_id, record_id)
        assert removed is True

        # Verify removal
        updated_nb = await notebook_manager.get_notebook(nb_id)
        assert len(updated_nb["records"]) == 0

    @pytest.mark.asyncio
    async def test_get_statistics(self, notebook_manager: NotebookManager) -> None:
        """Verify statistics aggregation."""
        await notebook_manager.ensure_initialized()

        nb1 = await notebook_manager.create_notebook("NB1")
        nb2 = await notebook_manager.create_notebook("NB2")

        await notebook_manager.add_record(
            [nb1["id"]],
            RecordType.SOLVE,
            "Q1",
            "Query",
            "Output",
        )
        await notebook_manager.add_record(
            [nb2["id"]],
            RecordType.QUESTION,
            "Q2",
            "Query",
            "Output",
        )

        stats = await notebook_manager.get_statistics()

        assert stats["total_notebooks"] == 2
        assert stats["total_records"] == 2
        assert stats["records_by_type"]["solve"] == 1
        assert stats["records_by_type"]["question"] == 1

    @pytest.mark.asyncio
    async def test_update_notebook(self, notebook_manager: NotebookManager) -> None:
        """Verify notebook metadata updates."""
        await notebook_manager.ensure_initialized()

        nb = await notebook_manager.create_notebook("Original", color="#0000FF")
        nb_id = nb["id"]

        updated = await notebook_manager.update_notebook(
            nb_id,
            name="Updated",
            color="#FF0000",
        )

        assert updated["name"] == "Updated"
        assert updated["color"] == "#FF0000"

        # Verify persistence
        reloaded = await notebook_manager.get_notebook(nb_id)
        assert reloaded["name"] == "Updated"

    @pytest.mark.asyncio
    async def test_delete_notebook(self, notebook_manager: NotebookManager) -> None:
        """Verify notebook deletion."""
        await notebook_manager.ensure_initialized()

        nb = await notebook_manager.create_notebook("To Delete")
        nb_id = nb["id"]

        deleted = await notebook_manager.delete_notebook(nb_id)
        assert deleted is True

        # Verify deletion
        reloaded = await notebook_manager.get_notebook(nb_id)
        assert reloaded is None
