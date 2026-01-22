"""Business logic for notebook operations.

Handles high-level operations on notebooks and records, including:
- CRUD operations on notebooks
- Record management within notebooks
- Statistics and aggregations
"""

import time
import uuid

from .notebook_models import (
    DEFAULT_COLOR,
    DEFAULT_ICON,
    RECENT_NOTEBOOKS_LIMIT,
    UUID_LENGTH,
    NotebookSummary,
    RecordType,
)
from .notebook_storage import NotebookStorage


class NotebookOperations:
    """High-level operations on notebooks and records.

    Uses dependency injection for storage layer to enable testing and
    future swappability (e.g., database backend).

    Attributes:
        storage: Storage layer for file I/O operations.
    """

    def __init__(self, storage: NotebookStorage) -> None:
        """Initialize operations with storage dependency.

        Args:
            storage: Storage layer instance for file operations.
        """
        self.storage = storage

    # === Notebook CRUD Operations ===

    async def create_notebook(
        self,
        name: str,
        description: str = "",
        color: str = DEFAULT_COLOR,
        icon: str = DEFAULT_ICON,
    ) -> dict:
        """Create a new notebook.

        Args:
            name: Human-readable notebook name (required).
            description: Optional notebook description.
            color: Hex color code for UI (default: blue).
            icon: Icon identifier for UI (default: "book").

        Returns:
            Created notebook as dictionary containing id, metadata, and empty
            records list.
        """
        notebook_id = str(uuid.uuid4())[:UUID_LENGTH]
        now = time.time()

        notebook = {
            "id": notebook_id,
            "name": name,
            "description": description,
            "created_at": now,
            "updated_at": now,
            "records": [],
            "color": color,
            "icon": icon,
        }

        # Persist notebook and update index
        await self.storage.save_notebook(notebook)
        await self._update_index_add_notebook(notebook)

        return notebook

    async def list_notebooks(self) -> list[NotebookSummary]:
        """List all notebooks with summary information.

        Retrieves summary data from index, sorted by most recently updated.
        Does not load full record lists (use get_notebook for that).

        Returns:
            List of NotebookSummary objects sorted by updated_at descending.
        """
        index = await self.storage.load_index()
        summaries: list[NotebookSummary] = []

        for nb_info in index.get("notebooks", []):
            # Verify notebook file still exists and has current data
            notebook = await self.storage.load_notebook(nb_info["id"])
            if notebook:
                summary = NotebookSummary(
                    id=notebook["id"],
                    name=notebook["name"],
                    description=notebook.get("description", ""),
                    created_at=notebook["created_at"],
                    updated_at=notebook["updated_at"],
                    record_count=len(notebook.get("records", [])),
                    color=notebook.get("color", DEFAULT_COLOR),
                    icon=notebook.get("icon", DEFAULT_ICON),
                )
                summaries.append(summary)

        # Sort by update time (most recent first)
        summaries.sort(key=lambda s: s.updated_at, reverse=True)
        return summaries

    async def get_notebook(self, notebook_id: str) -> dict | None:
        """Retrieve a complete notebook with all records.

        Args:
            notebook_id: Unique notebook identifier.

        Returns:
            Complete notebook dictionary, or None if not found.
        """
        return await self.storage.load_notebook(notebook_id)

    async def update_notebook(
        self,
        notebook_id: str,
        name: str | None = None,
        description: str | None = None,
        color: str | None = None,
        icon: str | None = None,
    ) -> dict | None:
        """Update notebook metadata.

        Only updates fields that are explicitly provided (None means skip).
        Preserves records and timestamps.

        Args:
            notebook_id: Unique notebook identifier.
            name: New notebook name (optional).
            description: New description (optional).
            color: New color code (optional).
            icon: New icon (optional).

        Returns:
            Updated notebook dictionary, or None if notebook not found.
        """
        notebook = await self.storage.load_notebook(notebook_id)
        if not notebook:
            return None

        # Update only provided fields
        if name is not None:
            notebook["name"] = name
        if description is not None:
            notebook["description"] = description
        if color is not None:
            notebook["color"] = color
        if icon is not None:
            notebook["icon"] = icon

        # Update timestamp
        notebook["updated_at"] = time.time()

        # Persist changes
        await self.storage.save_notebook(notebook)
        await self._update_index_modify_notebook(notebook)

        return notebook

    async def delete_notebook(self, notebook_id: str) -> bool:
        """Delete a notebook and all its records.

        Args:
            notebook_id: Unique notebook identifier.

        Returns:
            True if deleted successfully, False if not found.
        """
        deleted = await self.storage.delete_notebook(notebook_id)
        if deleted:
            await self._update_index_remove_notebook(notebook_id)

        return deleted

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
            title: Human-readable title.
            user_query: Original user input.
            output: Result/output from operation.
            metadata: Optional arbitrary metadata.
            kb_name: Optional knowledge base name.

        Returns:
            Dictionary containing:
                - record: The created record
                - added_to_notebooks: List of notebook IDs where added
        """
        record_id = str(uuid.uuid4())[:UUID_LENGTH]
        now = time.time()

        record = {
            "id": record_id,
            "type": record_type,
            "title": title,
            "user_query": user_query,
            "output": output,
            "metadata": metadata or {},
            "created_at": now,
            "kb_name": kb_name,
        }

        added_to = []
        for notebook_id in notebook_ids:
            notebook = await self.storage.load_notebook(notebook_id)
            if notebook:
                notebook["records"].append(record)
                notebook["updated_at"] = now

                await self.storage.save_notebook(notebook)
                await self._update_index_modify_notebook(notebook)
                added_to.append(notebook_id)

        return {"record": record, "added_to_notebooks": added_to}

    async def remove_record(self, notebook_id: str, record_id: str) -> bool:
        """Remove a record from a notebook.

        Args:
            notebook_id: Unique notebook identifier.
            record_id: Unique record identifier.

        Returns:
            True if record was removed, False if record or notebook not found.
        """
        notebook = await self.storage.load_notebook(notebook_id)
        if not notebook:
            return False

        original_count = len(notebook["records"])
        notebook["records"] = [r for r in notebook["records"] if r["id"] != record_id]

        if len(notebook["records"]) == original_count:
            return False  # Record not found

        notebook["updated_at"] = time.time()

        await self.storage.save_notebook(notebook)
        await self._update_index_modify_notebook(notebook)

        return True

    # === Statistics ===

    async def get_statistics(self) -> dict:
        """Get aggregate statistics across all notebooks.

        Returns:
            Dictionary containing:
                - total_notebooks: Count of all notebooks
                - total_records: Sum of all records across notebooks
                - records_by_type: Count per record type
                - recent_notebooks: List of recently updated notebooks
        """
        summaries = await self.list_notebooks()

        total_records = 0
        type_counts = {
            "solve": 0,
            "question": 0,
            "research": 0,
            "co_writer": 0,
        }

        for summary in summaries:
            notebook = await self.storage.load_notebook(summary.id)
            if notebook:
                for record in notebook.get("records", []):
                    total_records += 1
                    record_type = record.get("type", "")
                    if record_type in type_counts:
                        type_counts[record_type] += 1

        return {
            "total_notebooks": len(summaries),
            "total_records": total_records,
            "records_by_type": type_counts,
            "recent_notebooks": [s.model_dump() for s in summaries[:RECENT_NOTEBOOKS_LIMIT]],
        }

    # === Index Management (Private) ===

    async def _update_index_add_notebook(self, notebook: dict) -> None:
        """Add notebook to index file.

        Args:
            notebook: Notebook dictionary containing id, metadata.
        """
        index = await self.storage.load_index()
        index["notebooks"].append(
            {
                "id": notebook["id"],
                "name": notebook["name"],
                "description": notebook.get("description", ""),
                "created_at": notebook["created_at"],
                "updated_at": notebook["updated_at"],
                "record_count": 0,
                "color": notebook.get("color", DEFAULT_COLOR),
                "icon": notebook.get("icon", DEFAULT_ICON),
            }
        )
        await self.storage.save_index(index)

    async def _update_index_modify_notebook(self, notebook: dict) -> None:
        """Update notebook in index file.

        Args:
            notebook: Notebook dictionary with updated metadata and records.
        """
        index = await self.storage.load_index()
        for nb_info in index["notebooks"]:
            if nb_info["id"] == notebook["id"]:
                nb_info["name"] = notebook.get("name", nb_info["name"])
                nb_info["description"] = notebook.get("description", nb_info["description"])
                nb_info["color"] = notebook.get("color", nb_info["color"])
                nb_info["icon"] = notebook.get("icon", nb_info["icon"])
                nb_info["updated_at"] = notebook["updated_at"]
                nb_info["record_count"] = len(notebook.get("records", []))
                break

        await self.storage.save_index(index)

    async def _update_index_remove_notebook(self, notebook_id: str) -> None:
        """Remove notebook from index file.

        Args:
            notebook_id: Unique notebook identifier to remove.
        """
        index = await self.storage.load_index()
        index["notebooks"] = [nb for nb in index["notebooks"] if nb["id"] != notebook_id]
        await self.storage.save_index(index)
