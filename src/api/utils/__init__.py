"""API utilities package.

Provides utilities for API operations including notebook management, history tracking,
logging integration, and async task management.
"""

from .notebook_manager import NotebookManager, notebook_manager
from .notebook_models import (
    DEFAULT_COLOR,
    DEFAULT_ICON,
    JSON_INDENT,
    RECENT_NOTEBOOKS_LIMIT,
    UUID_LENGTH,
    Notebook,
    NotebookRecord,
    NotebookSummary,
    RecordType,
)
from .notebook_storage import NotebookStorage, NotebookStorageError

__all__ = [
    # Manager
    "NotebookManager",
    "notebook_manager",
    # Storage
    "NotebookStorage",
    "NotebookStorageError",
    # Models
    "Notebook",
    "NotebookRecord",
    "NotebookSummary",
    "RecordType",
    # Constants
    "JSON_INDENT",
    "UUID_LENGTH",
    "RECENT_NOTEBOOKS_LIMIT",
    "DEFAULT_COLOR",
    "DEFAULT_ICON",
]
