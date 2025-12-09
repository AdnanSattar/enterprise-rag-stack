"""
PII (Personally Identifiable Information) redaction.

Removes or masks sensitive information from documents before indexing.
"""

import re
from typing import Dict, List, Optional


class PIIRedactor:
    """
    Redact PII from text using regex patterns.

    Supports:
    - Email addresses
    - Phone numbers
    - SSNs
    - Credit card numbers
    - IP addresses
    """

    # Common PII patterns
    PATTERNS = {
        "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "phone": r"(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}",
        "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
        "credit_card": r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",
        "ip_address": r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b",
    }

    def __init__(self, custom_patterns: Optional[Dict[str, str]] = None):
        """
        Args:
            custom_patterns: Additional regex patterns to use
        """
        self.patterns = {**self.PATTERNS}
        if custom_patterns:
            self.patterns.update(custom_patterns)

    def redact(
        self, text: str, replacement: str = "[REDACTED]"
    ) -> tuple[str, List[str]]:
        """
        Redact PII from text.

        Args:
            text: Input text
            replacement: String to replace PII with

        Returns:
            Tuple of (redacted_text, list_of_redacted_items)
        """
        redacted_text = text
        redacted_items = []

        for pii_type, pattern in self.patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                redacted_items.append(
                    {"type": pii_type, "value": match.group(), "position": match.span()}
                )
                redacted_text = redacted_text.replace(match.group(), replacement)

        return redacted_text, redacted_items

    def detect(self, text: str) -> List[Dict]:
        """
        Detect PII without redacting.

        Args:
            text: Input text

        Returns:
            List of detected PII items
        """
        detected = []

        for pii_type, pattern in self.patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                detected.append(
                    {"type": pii_type, "value": match.group(), "position": match.span()}
                )

        return detected


def redact_pii(text: str, replacement: str = "[REDACTED]") -> tuple[str, List[str]]:
    """
    Convenience function for PII redaction.

    Args:
        text: Input text
        replacement: Replacement string

    Returns:
        Tuple of (redacted_text, redacted_items)
    """
    redactor = PIIRedactor()
    return redactor.redact(text, replacement)
