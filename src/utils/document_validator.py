#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Document Validator - Validation utilities for document uploads
"""

import os
import re


class DocumentValidator:
    """Document validation utilities"""

    # Maximum file size in bytes (100MB)
    MAX_FILE_SIZE = 100 * 1024 * 1024

    # Allowed file extensions
    ALLOWED_EXTENSIONS = {
        ".pdf",
        ".txt",
        ".md",
        ".doc",
        ".docx",
        ".rtf",
        ".html",
        ".htm",
        ".xml",
        ".json",
        ".csv",
        ".xlsx",
        ".xls",
        ".pptx",
        ".ppt",
    }

    @staticmethod
    def validate_upload_safety(
        filename: str, file_size: int, allowed_extensions: set[str] | None = None
    ) -> str:
        """
        Validate file upload safety

        Args:
            filename: Name of the file
            file_size: Size of the file in bytes
            allowed_extensions: Optional override for allowed extensions

        Returns:
            Sanitized filename safe for filesystem use

        Raises:
            ValueError: If validation fails
        """
        # Check file size
        if file_size > DocumentValidator.MAX_FILE_SIZE:
            raise ValueError(
                f"File too large: {file_size} bytes. Maximum allowed: {DocumentValidator.MAX_FILE_SIZE} bytes"
            )

        # Sanitize filename - remove path components and dangerous characters
        # Extract just the filename, removing any path components
        safe_name = os.path.basename(filename)
        # Remove null bytes and other control characters
        safe_name = re.sub(r"[\x00-\x1f\x7f]", "", safe_name)
        # Replace problematic characters
        safe_name = re.sub(r'[<>:"/\\|?*]', "_", safe_name)

        if not safe_name or safe_name in (".", "..") or safe_name.strip("_") == "":
            raise ValueError("Invalid filename")

        # Check file extension
        _, ext = os.path.splitext(safe_name.lower())
        exts_to_check = allowed_extensions or DocumentValidator.ALLOWED_EXTENSIONS
        if ext not in exts_to_check:
            raise ValueError(
                f"Unsupported file type: {ext}. Allowed types: {', '.join(exts_to_check)}"
            )

        return safe_name

    @staticmethod
    def get_file_info(filename: str, file_size: int) -> dict:
        """
        Get file information

        Args:
            filename: Name of the file
            file_size: Size of the file in bytes

        Returns:
            Dictionary with file information
        """
        _, ext = os.path.splitext(filename.lower())
        return {
            "filename": filename,
            "extension": ext,
            "size_bytes": file_size,
            "size_mb": round(file_size / (1024 * 1024), 2),
            "is_allowed": ext in DocumentValidator.ALLOWED_EXTENSIONS,
        }

    @staticmethod
    def validate_file(path: str) -> dict:
        """
        Validate that a file exists, is readable, and has valid content.

        Args:
            path: Path to the file to validate

        Returns:
            File info dictionary

        Raises:
            ValueError: If file is missing or validation fails
        """
        if not os.path.exists(path):
            raise ValueError(f"File not found: {path}")

        if not os.path.isfile(path):
            raise ValueError(f"Not a file: {path}")

        if not os.access(path, os.R_OK):
            raise ValueError(f"File not readable: {path}")

        size = os.path.getsize(path)
        filename = os.path.basename(path)

        # Validate using validate_upload_safety
        safe_name = DocumentValidator.validate_upload_safety(filename, size)

        return DocumentValidator.get_file_info(safe_name, size)
