"""
Document Text Extractor - Extract text from various document formats.
"""

import os
from typing import Optional


class DocumentTextExtractor:
    """Extract text content from various document formats."""

    @staticmethod
    def extract_text(file_path: str) -> Optional[str]:
        """
        Extract text from a file based on its extension.

        Args:
            file_path: Path to the file

        Returns:
            Extracted text content, or None if extraction fails
        """
        if not os.path.exists(file_path):
            return None

        ext = os.path.splitext(file_path)[1].lower()

        try:
            if ext == '.pdf':
                return DocumentTextExtractor._extract_pdf(file_path)
            elif ext == '.docx':
                return DocumentTextExtractor._extract_docx(file_path)
            elif ext == '.txt':
                return DocumentTextExtractor._extract_txt(file_path)
            else:
                return None
        except (OSError, ValueError, ImportError):
            return None

    @staticmethod
    def _extract_pdf(file_path: str) -> str:
        """Extract text from PDF file."""
        try:
            import PyPDF2
        except ImportError:
            raise ImportError("PyPDF2 required for PDF extraction. Install with: pip install PyPDF2")

        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            text = ''
            for page in reader.pages:
                text += page.extract_text() + '\n'
            return text.strip()

    @staticmethod
    def _extract_docx(file_path: str) -> str:
        """Extract text from DOCX file."""
        try:
            from docx import Document
        except ImportError:
            raise ImportError("python-docx required for DOCX extraction. Install with: pip install python-docx")

        doc = Document(file_path)
        text = ''
        for para in doc.paragraphs:
            text += para.text + '\n'
        return text.strip()

    @staticmethod
    def _extract_txt(file_path: str) -> str:
        """Extract text from TXT file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
