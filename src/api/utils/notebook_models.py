"""Data models for notebook system.

Defines all Pydantic models and enums used for notebook management,
including notebooks, records, and metadata structures.
"""

from enum import Enum

from pydantic import BaseModel, Field

# Constants
JSON_INDENT: int = 2
UUID_LENGTH: int = 8
RECENT_NOTEBOOKS_LIMIT: int = 5
DEFAULT_COLOR: str = "#3B82F6"
DEFAULT_ICON: str = "book"


class RecordType(str, Enum):
    """Enumeration of supported record types."""

    SOLVE = "solve"
    QUESTION = "question"
    RESEARCH = "research"
    CO_WRITER = "co_writer"


class NotebookRecord(BaseModel):
    """Single record stored within a notebook.

    Attributes:
        id: Unique record identifier.
        type: Type of record (from RecordType enum).
        title: Human-readable record title.
        user_query: Original user input/query that generated this record.
        output: The output/result from the operation.
        metadata: Optional arbitrary metadata associated with the record.
        created_at: Unix timestamp of record creation.
        kb_name: Optional knowledge base name associated with the record.
    """

    id: str
    type: RecordType
    title: str
    user_query: str
    output: str
    metadata: dict = Field(default_factory=dict)
    created_at: float
    kb_name: str | None = None


class Notebook(BaseModel):
    """Container for notebook records and metadata.

    Attributes:
        id: Unique notebook identifier (8-character UUID).
        name: Human-readable notebook name.
        description: Optional notebook description.
        created_at: Unix timestamp of notebook creation.
        updated_at: Unix timestamp of last update.
        records: List of records stored in this notebook.
        color: Hex color code for UI display.
        icon: Icon identifier for UI display.
    """

    id: str
    name: str
    description: str = ""
    created_at: float
    updated_at: float
    records: list[NotebookRecord] = Field(default_factory=list)
    color: str = DEFAULT_COLOR
    icon: str = DEFAULT_ICON


class NotebookSummary(BaseModel):
    """Summary information for a notebook (used in list operations).

    Provides lightweight metadata for notebooks without including full record list.

    Attributes:
        id: Unique notebook identifier.
        name: Notebook name.
        description: Notebook description.
        created_at: Creation timestamp.
        updated_at: Last update timestamp.
        record_count: Number of records in notebook.
        color: UI display color.
        icon: UI display icon.
    """

    id: str
    name: str
    description: str
    created_at: float
    updated_at: float
    record_count: int
    color: str
    icon: str
