# -*- coding: utf-8 -*-
"""
Text Parser
===========

Parser for plain text documents (.txt files).
"""

from pathlib import Path
from typing import Any

from ...types import Document
from ..base import BaseComponent


class TextParser(BaseComponent):
    """
    Plain text parser.

    Parses text files (.txt) into Document objects.
    Also handles common text-based formats.
    """

    name = "text_parser"

    # Supported extensions
    SUPPORTED_EXTENSIONS = {".txt", ".text", ".log", ".csv", ".tsv"}

    async def process(self, data: str | Path, **kwargs: Any) -> Document:
        """
        Parse a text file into a Document.

        Args:
            data: Path to the text file (str or Path)
            **kwargs: Additional arguments

        Returns:
            Parsed Document
        """
        if not isinstance(data, (str, Path)):
            raise TypeError(f"Expected str or Path, got {type(data).__name__}")

        file_path = Path(data)

        if not file_path.exists():
            raise FileNotFoundError(f"Text file not found: {file_path}")

        self.logger.info(f"Parsing text file: {file_path.name}")

        # Try different encodings
        content = None
        encodings = ["utf-8", "utf-8-sig", "gbk", "gb2312", "latin-1"]

        for encoding in encodings:
            try:
                with open(file_path, encoding=encoding) as f:
                    content = f.read()
            except UnicodeDecodeError:
                self.logger.warning(
                    f"Unicode decode error for file {file_path}, trying next encoding."
                )
                continue
            except OSError as e:
                self.logger.error(f"Failed to read file {file_path}: {e}")
                raise

        if content is None:
            # Last resort: read as binary and decode with error handling
            try:
                with open(file_path, "rb") as f:
                    content = f.read().decode("utf-8", errors="replace")
            except OSError as e:
                self.logger.error(f"Failed to read file {file_path} as binary: {e}")
                raise

        return Document(
            content=content,
            file_path=str(file_path),
            metadata={
                "filename": file_path.name,
                "parser": self.name,
                "extension": file_path.suffix.lower(),
                "size_bytes": file_path.stat().st_size,
            },
        )

    @classmethod
    def can_parse(cls, file_path: str | Path) -> bool:
        """
        Check if this parser can handle the given file.

        Args:
            file_path: Path to check

        Returns:
            True if file can be parsed
        """
        suffix = Path(file_path).suffix.lower()
        return suffix in cls.SUPPORTED_EXTENSIONS
