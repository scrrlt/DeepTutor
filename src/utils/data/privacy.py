"""
PII Firewall - Scan and redact sensitive information from prompts.
"""

import re
from typing import Dict


class PIIFirewall:
    """Firewall for detecting and redacting PII in text."""

    def __init__(self):
        # Patterns from pii_scanner.py and pii_redaction.py
        self.patterns = {
            "email": r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
            "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
            "digits": r"\b\d{10,}\b",  # Phone numbers, etc.
        }

    def scan(self, text: str) -> Dict[str, int]:
        """Scan text for PII occurrences."""
        results = {}
        for p_type, pattern in self.patterns.items():
            results[p_type] = len(re.findall(pattern, text))
        return results

    def sanitize(self, text: str) -> str:
        """Redact PII from text."""
        for p_type, pattern in self.patterns.items():
            text = re.sub(pattern, f"<{p_type.upper()}_REDACTED>", text)
        return text
