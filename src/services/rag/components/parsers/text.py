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
        Parse a plain-text file from the given path into a Document object.
        
        Attempts to read the file using a sequence of common text encodings and, if those fail, falls back to a binary read decoded with UTF-8 using replacement for invalid bytes. The returned Document contains the file content and metadata including filename, parser name, file extension, and size in bytes.
        
        Parameters:
            data (str | Path): Path to the text file to parse.
            **kwargs: Additional unused keyword arguments.
        
        Returns:
            Document: A Document containing the file content and metadata:
                - content: the text content of the file
                - file_path: stringified file path
                - metadata: dict with keys `filename`, `parser`, `extension`, and `size_bytes`
        
        Raises:
            TypeError: If `data` is not a str or Path.
            FileNotFoundError: If the specified file does not exist.
            OSError: If an I/O error occurs while reading the file.
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
                with open(file_path, "r", encoding=encoding) as f:
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
        Determine whether this parser supports the file's extension.
        
        @returns: `True` if the file's extension is one of the parser's supported extensions, `False` otherwise.
        """
        suffix = Path(file_path).suffix.lower()
        return suffix in cls.SUPPORTED_EXTENSIONS