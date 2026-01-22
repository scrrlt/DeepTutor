"""File storage layer for notebook system.

Handles all file I/O operations for notebook persistence, including:
- Index file management (list of all notebooks)
- Notebook file loading/saving
- Thread pool offloading for CPU-bound JSON operations
"""

import asyncio
import json
from pathlib import Path

import aiofiles

from .notebook_models import JSON_INDENT


class NotebookStorageError(Exception):
    """Base exception for storage layer errors."""


class NotebookStorage:
    """Manages file-based persistence for notebooks.

    Provides async file I/O with thread pool optimization for JSON serialization
    to prevent event loop blocking on large data structures.

    Attributes:
        base_dir: Root directory for notebook storage.
        index_file: Path to the index file listing all notebooks.
    """

    def __init__(self, base_dir: Path) -> None:
        """Initialize storage with base directory.

        Args:
            base_dir: Directory path for storing notebook files.

        Raises:
            NotebookStorageError: If base directory cannot be created.
        """
        self.base_dir = base_dir
        self.index_file = base_dir / "notebooks_index.json"

        try:
            self.base_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            msg = f"Failed to create notebook storage directory: {e}"
            raise NotebookStorageError(msg) from e

    async def ensure_index_exists(self) -> None:
        """Ensure index file exists, creating it if necessary.

        Creates an empty index with `{"notebooks": []}` structure if file
        does not exist.

        Raises:
            NotebookStorageError: If index file cannot be created.
        """
        if self.index_file.exists():
            return

        try:
            content = json.dumps(
                {"notebooks": []}, indent=JSON_INDENT, ensure_ascii=False
            )
            async with aiofiles.open(self.index_file, "w", encoding="utf-8") as f:
                await f.write(content)
        except OSError as e:
            msg = f"Failed to create index file: {e}"
            raise NotebookStorageError(msg) from e

    async def load_index(self) -> dict:
        """Load and parse the notebook index file.

        JSON parsing is offloaded to thread pool to prevent event loop blocking
        on large index files.

        Returns:
            Dictionary with structure: `{"notebooks": [...]}`

        Raises:
            NotebookStorageError: If index file cannot be read.
        """
        try:
            async with aiofiles.open(self.index_file, encoding="utf-8") as f:
                content = await f.read()

            # Offload JSON parsing to thread pool to avoid blocking event loop
            loop = asyncio.get_running_loop()
            index: dict = await loop.run_in_executor(None, json.loads, content)
            return index
        except (OSError, json.JSONDecodeError) as e:
            # Return empty index on read/parse error
            if isinstance(e, OSError):
                return {"notebooks": []}
            msg = f"Failed to parse index file: {e}"
            raise NotebookStorageError(msg) from e

    async def save_index(self, index: dict) -> None:
        """Save index to file.

        JSON serialization is offloaded to thread pool to prevent event loop blocking.

        Args:
            index: Index dictionary to save.

        Raises:
            NotebookStorageError: If index file cannot be written.
        """
        try:
            loop = asyncio.get_running_loop()
            content: str = await loop.run_in_executor(
                None,
                lambda: json.dumps(index, indent=JSON_INDENT, ensure_ascii=False),
            )

            async with aiofiles.open(self.index_file, "w", encoding="utf-8") as f:
                await f.write(content)
        except OSError as e:
            msg = f"Failed to save index file: {e}"
            raise NotebookStorageError(msg) from e

    def _get_notebook_path(self, notebook_id: str) -> Path:
        """Get file path for a notebook.

        Args:
            notebook_id: Unique notebook identifier.

        Returns:
            Path object for the notebook file.
        """
        return self.base_dir / f"{notebook_id}.json"

    async def load_notebook(self, notebook_id: str) -> dict | None:
        """Load a notebook from file.

        JSON parsing is offloaded to thread pool to prevent event loop blocking
        on large notebooks (5MB+).

        Args:
            notebook_id: Unique notebook identifier.

        Returns:
            Notebook dictionary, or None if file not found.

        Raises:
            NotebookStorageError: If notebook cannot be read (non-404 errors).
        """
        filepath = self._get_notebook_path(notebook_id)

        if not filepath.exists():
            return None

        try:
            async with aiofiles.open(filepath, encoding="utf-8") as f:
                content = await f.read()

            # Offload JSON parsing to thread pool
            loop = asyncio.get_running_loop()
            notebook: dict = await loop.run_in_executor(None, json.loads, content)
            return notebook
        except json.JSONDecodeError as e:
            msg = f"Failed to parse notebook {notebook_id}: {e}"
            raise NotebookStorageError(msg) from e
        except OSError as e:
            msg = f"Failed to read notebook {notebook_id}: {e}"
            raise NotebookStorageError(msg) from e

    async def save_notebook(self, notebook: dict) -> None:
        """Save notebook to file.

        JSON serialization is offloaded to thread pool to prevent event loop blocking
        on large notebooks.

        Args:
            notebook: Notebook dictionary with 'id' key.

        Raises:
            NotebookStorageError: If notebook cannot be written.
            KeyError: If notebook dict missing 'id' key.
        """
        try:
            notebook_id = notebook["id"]
            filepath = self._get_notebook_path(notebook_id)

            loop = asyncio.get_running_loop()
            content: str = await loop.run_in_executor(
                None,
                lambda: json.dumps(notebook, indent=JSON_INDENT, ensure_ascii=False),
            )

            async with aiofiles.open(filepath, "w", encoding="utf-8") as f:
                await f.write(content)
        except OSError as e:
            msg = f"Failed to save notebook: {e}"
            raise NotebookStorageError(msg) from e

    async def delete_notebook(self, notebook_id: str) -> bool:
        """Delete notebook file.

        Args:
            notebook_id: Unique notebook identifier.

        Returns:
            True if file was deleted, False if file did not exist.

        Raises:
            NotebookStorageError: If file deletion fails (non-404 errors).
        """
        filepath = self._get_notebook_path(notebook_id)

        if not filepath.exists():
            return False

        try:
            filepath.unlink()
            return True
        except OSError as e:
            msg = f"Failed to delete notebook {notebook_id}: {e}"
            raise NotebookStorageError(msg) from e
