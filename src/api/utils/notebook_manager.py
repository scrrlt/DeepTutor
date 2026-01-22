"""Unified notebook manager orchestrating storage and operations layers.

Provides a simple, high-level API for notebook management by coordinating
the storage and operations layers. Acts as the main entry point for consumers.
"""

from pathlib import Path

from .notebook_models import RecordType
from .notebook_operations import NotebookOperations
from .notebook_storage import NotebookStorage


class NotebookManager:
    """High-level notebook management API.

    Orchestrates storage and operations layers to provide a clean, unified
    interface for notebook management. Uses lazy initialization of the
    operations layer.

    Attributes:
        storage: File storage layer instance.
        _operations: Business logic layer instance (lazily initialized).
    """

    def __init__(self, base_dir: str | None = None) -> None:
        """Initialize notebook manager with storage directory.

        Args:
            base_dir: Base directory for notebook storage. If None, defaults to
                `<project_root>/data/user/notebook`.

        Raises:
            NotebookStorageError: If storage directory cannot be created.
        """
        if base_dir is None:
            # Compute project root from file location:
            # src/api/utils/notebook_manager.py -> 3 levels up to project root
            project_root = Path(__file__).resolve().parents[3]
            base_dir_path = project_root / "data" / "user" / "notebook"
        else:
            base_dir_path = Path(base_dir)

        self.storage = NotebookStorage(base_dir_path)
        self._operations: NotebookOperations | None = None

    def _get_operations(self) -> NotebookOperations:
        """Get or create operations layer (lazy initialization).

        Returns:
            NotebookOperations instance.
        """
        if self._operations is None:
            self._operations = NotebookOperations(self.storage)

        return self._operations

    async def ensure_initialized(self) -> None:
        """Ensure storage is initialized (creates index if needed).

        Should be called during application startup to guarantee the index
        file exists before first use.
        """
        await self.storage.ensure_index_exists()

    # === Notebook Operations ===

    async def create_notebook(
        self,
        name: str,
        description: str = "",
        color: str = "#3B82F6",
        icon: str = "book",
    ) -> dict:
        """Create a new notebook.

        Args:
            name: Human-readable notebook name.
            description: Optional notebook description.
            color: Hex color code for UI (default: blue).
            icon: Icon identifier for UI (default: "book").

        Returns:
            Created notebook as dictionary.

        Raises:
            NotebookStorageError: If notebook creation fails.
        """
        ops = self._get_operations()
        return await ops.create_notebook(name, description, color, icon)

    async def list_notebooks(self) -> list[dict]:
        """List all notebooks with summary information.

        Returns:
            List of notebook summaries sorted by update time (descending).

        Raises:
            NotebookStorageError: If listing fails.
        """
        ops = self._get_operations()
        summaries = await ops.list_notebooks()
        return [s.model_dump() for s in summaries]

    async def get_notebook(self, notebook_id: str) -> dict | None:
        """Retrieve a complete notebook with all records.

        Args:
            notebook_id: Unique notebook identifier.

        Returns:
            Complete notebook dictionary, or None if not found.

        Raises:
            NotebookStorageError: If retrieval fails.
        """
        ops = self._get_operations()
        return await ops.get_notebook(notebook_id)

    async def update_notebook(
        self,
        notebook_id: str,
        name: str | None = None,
        description: str | None = None,
        color: str | None = None,
        icon: str | None = None,
    ) -> dict | None:
        """Update notebook metadata.

        Args:
            notebook_id: Unique notebook identifier.
            name: New notebook name (optional).
            description: New description (optional).
            color: New color code (optional).
            icon: New icon (optional).

        Returns:
            Updated notebook dictionary, or None if not found.

        Raises:
            NotebookStorageError: If update fails.
        """
        ops = self._get_operations()
        return await ops.update_notebook(notebook_id, name, description, color, icon)

    async def delete_notebook(self, notebook_id: str) -> bool:
        """Delete a notebook and all its records.

        Args:
            notebook_id: Unique notebook identifier.

        Returns:
            True if deleted, False if not found.

        Raises:
            NotebookStorageError: If deletion fails.
        """
        ops = self._get_operations()
        return await ops.delete_notebook(notebook_id)

    # === Record Operations ===

    async def add_record(
        self,
        notebook_ids: list[str],
        record_type: RecordType,
        title: str,
        user_query: str,
        output: str,
        metadata: dict | None = None,
        kb_name: str | None = None,
    ) -> dict:
        """Add a record to one or more notebooks.

        Args:
            notebook_ids: List of target notebook IDs.
            record_type: Type of record (from RecordType enum).
            title: Human-readable record title.
            user_query: Original user input.
            output: Result/output from operation.
            metadata: Optional arbitrary metadata.
            kb_name: Optional knowledge base name.

        Returns:
            Dictionary with 'record' and 'added_to_notebooks' keys.

        Raises:
            NotebookStorageError: If operation fails.
        """
        ops = self._get_operations()
        return await ops.add_record(
            notebook_ids,
            record_type,
            title,
            user_query,
            output,
            metadata,
            kb_name,
        )

    async def remove_record(self, notebook_id: str, record_id: str) -> bool:
        """Remove a record from a notebook.

        Args:
            notebook_id: Unique notebook identifier.
            record_id: Unique record identifier.

        Returns:
            True if removed, False if not found.

        Raises:
            NotebookStorageError: If operation fails.
        """
        ops = self._get_operations()
        return await ops.remove_record(notebook_id, record_id)

    # === Statistics ===

    async def get_statistics(self) -> dict:
        """Get aggregate statistics across all notebooks.

        Returns:
            Dictionary with notebook and record counts and summaries.

        Raises:
            NotebookStorageError: If operation fails.
        """
        ops = self._get_operations()
        return await ops.get_statistics()


# Global singleton instance for backward compatibility
notebook_manager = NotebookManager()
