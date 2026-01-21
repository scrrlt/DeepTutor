# -*- coding: utf-8 -*-
"""
Markdown Parser
===============

Parser for Markdown documents.
"""

from pathlib import Path
from typing import Union

from ...types import Document
from ..base import BaseComponent


class MarkdownParser(BaseComponent):
    """
    Markdown parser.

    Parses Markdown files into Document objects.
    """

    name = "markdown_parser"

    async def process(self, file_path: Union[str, Path], **kwargs) -> Document:
        """
        Parse a Markdown file into a Document.
        
        Parameters:
            file_path (Union[str, Path]): Path to the Markdown file to parse.
        
        Returns:
            Document: Document with `content` set to the file text, `file_path` set to the path string, and `metadata` containing `"filename"` and `"parser"`.
        
        Raises:
            FileNotFoundError: If the specified path does not exist.
            OSError: If an error occurs while reading the file.
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Markdown file not found: {file_path}")

        self.logger.info(f"Parsing Markdown: {file_path.name}")

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        except OSError as e:
            self.logger.error(f"Failed to read file {file_path}: {e}")
            raise

        return Document(
            content=content,
            file_path=str(file_path),
            metadata={
                "filename": file_path.name,
                "parser": self.name,
            },
        )