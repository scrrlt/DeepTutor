# -*- coding: utf-8 -*-
"""
Document Validation Utilities
=============================

Input validation and sanitization for document processing.
"""

import logging
from pathlib import Path
import re
from typing import Set

logger = logging.getLogger(__name__)

# Supported document formats
SUPPORTED_EXTENSIONS: Set[str] = {
    ".pdf",
    ".md",
    ".txt",
    ".html",
    ".htm",
    ".docx",
    ".doc",
    ".rtf",
    ".odt",
}

# Maximum file size (100MB)
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB in bytes

# Constant for megabyte conversion
MB = 1024 * 1024

# Maximum content length for processing
MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10MB in characters

# ASCII character constants for filtering
ASCII_PRINTABLE_MIN = 32  # Space character
ASCII_PRINTABLE_MAX = 126  # Tilde character
ASCII_TAB = 9
ASCII_NEWLINE = 10


def is_printable_char(char: str) -> bool:
    """Return True if the character is printable (including newlines and tabs)."""
    if not char:
        return False
    code_point = ord(char)
    return (
        ASCII_PRINTABLE_MIN <= code_point <= ASCII_PRINTABLE_MAX
        or code_point in (ASCII_TAB, ASCII_NEWLINE)
    )


class DocumentValidator:
    """Document input validation and sanitization utilities."""

    @staticmethod
    def validate_file(file_path: Path) -> None:
        """
        Validate a document file before processing.

        Args:
            file_path: Path to the document file

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file is invalid (empty, too large, unsupported format)
            PermissionError: If file cannot be read
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Document file not found: {file_path}")

        if not file_path.is_file():
            raise ValueError(f"Path is not a file: {file_path}")

        # Check file size
        file_size = file_path.stat().st_size
        if file_size == 0:
            raise ValueError(f"Document file is empty: {file_path}")

        if file_size > MAX_FILE_SIZE:
            raise ValueError(
                f"Document file too large: {file_path} "
                f"({file_size / MB:.1f}MB > {MAX_FILE_SIZE / MB}MB)"
            )

        # Check file extension
        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported document format: {file_path.suffix}. "
                f"Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )

        # Check if file is readable
        try:
            with open(file_path, "rb") as f:
                f.read(1)  # Try to read one byte
        except (PermissionError, OSError) as e:
            raise PermissionError(f"Cannot read document file: {file_path}. Error: {e}")

        logger.debug(f"Document validation passed for: {file_path}")

    @staticmethod
    def sanitize_content(content: str) -> str:
        """
        Sanitize document content to remove problematic characters.

        Args:
            content: Raw document content

        Returns:
            Sanitized content
        """
        if not content:
            return content

        # Remove null bytes and other control characters (except newlines and tabs)
        sanitized = "".join(char for char in content if is_printable_char(char))

        # Limit content length
        if len(sanitized) > MAX_CONTENT_LENGTH:
            logger.warning(
                f"Content length {len(sanitized)} exceeds maximum {MAX_CONTENT_LENGTH}, "
                "truncating content."
            )
            sanitized = sanitized[:MAX_CONTENT_LENGTH]

        return sanitized.strip()

    @staticmethod
    def validate_upload_safety(filename: str, file_size_bytes: int, allowed_extensions: set[str] | None = None) -> str:
        """
        Validate filename and size before file upload/write.
        
        Args:
            filename: Original filename
            file_size_bytes: Size of file in bytes
            allowed_extensions: Set of allowed file extensions (e.g., {'.pdf', '.txt'}). 
                              If None, defaults to {'.pdf'} for backward compatibility.
            
        Returns:
            str: Sanitized filename
            
        Raises:
            ValueError: If filename or size is invalid
        """
        if allowed_extensions is None:
            allowed_extensions = {'.pdf'}
            
        # Check for null bytes
        if '\x00' in filename:
            raise ValueError("Filename contains null bytes")
            
        # Check filename length
        if len(filename) == 0:
            raise ValueError("Filename cannot be empty")
        if len(filename) > 255:  # Common filesystem limit
            raise ValueError("Filename too long (max 255 characters)")
            
        # Sanitize filename - remove path components and dangerous chars
        safe_name = Path(filename).name
        
        # Additional validation: only allow safe characters
        if not re.match(r'^[a-zA-Z0-9._-]+$', safe_name):
            raise ValueError("Filename contains invalid characters. Only alphanumeric, dots, underscores, and hyphens allowed")
            
        # Check file extension against allowed extensions
        file_ext = Path(safe_name).suffix.lower()
        if file_ext not in allowed_extensions:
            raise ValueError(
                f"Unsupported file type: {file_ext}. "
                f"Allowed types: {', '.join(sorted(allowed_extensions))}"
            )
            
        # Check file size
        if file_size_bytes == 0:
            raise ValueError("File cannot be empty")
        if file_size_bytes > MAX_FILE_SIZE:
            raise ValueError(
                f"File too large: {file_size_bytes / MB:.1f}MB > {MAX_FILE_SIZE / MB}MB"
            )
            
        return safe_name
