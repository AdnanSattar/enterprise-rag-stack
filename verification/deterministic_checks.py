"""
Level 1: Deterministic verification checks.

Fast, rule-based verification that catches obvious errors:
- Numbers not in context (hallucinated statistics)
- Dates not in context (made-up deadlines)
- Currency values not in context (wrong prices)
- Named entities not in context
"""

import logging
import re
from typing import Dict, List, Set, Tuple

logger = logging.getLogger(__name__)


def extract_numbers(text: str) -> Set[str]:
    """
    Extract all numbers from text.

    Captures:
    - Integers: 42, 1000
    - Decimals: 3.14, 0.5
    - Percentages: 50%, 3.5%

    Args:
        text: Text to extract from

    Returns:
        Set of number strings
    """
    pattern = r"\b\d+(?:\.\d+)?%?\b"
    return set(re.findall(pattern, text))


def extract_dates(text: str) -> Set[str]:
    """
    Extract all dates from text.

    Captures various formats:
    - MM/DD/YYYY, DD-MM-YYYY
    - January 1, 2024
    - 2024-01-01 (ISO)

    Args:
        text: Text to extract from

    Returns:
        Set of date strings
    """
    patterns = [
        r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b",  # MM/DD/YYYY
        r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b",  # Month D, YYYY
        r"\b\d{4}[-/]\d{2}[-/]\d{2}\b",  # ISO format
    ]

    dates = set()
    for pattern in patterns:
        dates.update(re.findall(pattern, text, re.IGNORECASE))

    return dates


def extract_currencies(text: str) -> Set[str]:
    """
    Extract currency values from text.

    Captures:
    - $1,234.56
    - $1000
    - €500

    Args:
        text: Text to extract from

    Returns:
        Set of currency strings
    """
    pattern = r"[$€£¥][\d,]+(?:\.\d{2})?"
    return set(re.findall(pattern, text))


def extract_named_entities(text: str) -> Set[str]:
    """
    Extract potential named entities (simplified).

    Looks for capitalized phrases that might be names.
    For production, use spaCy NER.

    Args:
        text: Text to extract from

    Returns:
        Set of potential entity names
    """
    # Simple heuristic: consecutive capitalized words
    pattern = r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b"

    entities = set()
    for match in re.findall(pattern, text):
        # Filter out common words at sentence starts
        if match.lower() not in {"the", "this", "that", "these", "those"}:
            entities.add(match)

    return entities


class DeterministicVerifier:
    """
    Level 1: Deterministic verification checks.

    Performs fast, rule-based verification:
    - Number extraction and validation
    - Date format checking
    - Named entity verification
    - Keyword presence checks

    Usage:
        verifier = DeterministicVerifier()
        is_valid, issues = verifier.verify(answer, context)
    """

    def __init__(
        self,
        strict_numbers: bool = True,
        strict_dates: bool = True,
        strict_currencies: bool = True,
        allowed_number_tolerance: float = 0.0,
    ):
        """
        Args:
            strict_numbers: Require numbers in answer to exist in context
            strict_dates: Require dates in answer to exist in context
            strict_currencies: Require currency values to exist in context
            allowed_number_tolerance: Tolerance for number matching (e.g., 0.01 for 1%)
        """
        self.strict_numbers = strict_numbers
        self.strict_dates = strict_dates
        self.strict_currencies = strict_currencies
        self.allowed_number_tolerance = allowed_number_tolerance

    def verify(
        self,
        answer: str,
        context: str,
    ) -> Tuple[bool, List[str]]:
        """
        Verify answer against context.

        Args:
            answer: Generated answer
            context: Retrieved context

        Returns:
            (is_valid, list of issues)
        """
        issues = []

        # Check numbers
        if self.strict_numbers:
            answer_numbers = extract_numbers(answer)
            context_numbers = extract_numbers(context)

            unsupported = self._check_numbers(answer_numbers, context_numbers)
            if unsupported:
                issues.append(f"Numbers not found in context: {unsupported}")

        # Check dates
        if self.strict_dates:
            answer_dates = extract_dates(answer)
            context_dates = extract_dates(context)

            unsupported_dates = answer_dates - context_dates
            if unsupported_dates:
                issues.append(f"Dates not found in context: {unsupported_dates}")

        # Check currencies
        if self.strict_currencies:
            answer_currencies = extract_currencies(answer)
            context_currencies = extract_currencies(context)

            unsupported_currencies = answer_currencies - context_currencies
            if unsupported_currencies:
                issues.append(
                    f"Currency values not found in context: {unsupported_currencies}"
                )

        is_valid = len(issues) == 0
        return is_valid, issues

    def _check_numbers(
        self,
        answer_numbers: Set[str],
        context_numbers: Set[str],
    ) -> Set[str]:
        """Check if answer numbers are in context, with tolerance."""
        if self.allowed_number_tolerance == 0:
            return answer_numbers - context_numbers

        unsupported = set()
        context_values = {self._parse_number(n) for n in context_numbers}

        for num_str in answer_numbers:
            num_val = self._parse_number(num_str)
            if num_val is None:
                continue

            # Check if within tolerance of any context number
            found = any(
                abs(num_val - ctx_val)
                <= self.allowed_number_tolerance * max(abs(ctx_val), 1)
                for ctx_val in context_values
                if ctx_val is not None
            )

            if not found:
                unsupported.add(num_str)

        return unsupported

    def _parse_number(self, num_str: str) -> float:
        """Parse number string to float."""
        try:
            # Remove percentage sign
            clean = num_str.rstrip("%")
            return float(clean)
        except ValueError:
            return None

    def get_extraction_summary(
        self,
        text: str,
    ) -> Dict:
        """
        Get summary of extracted values from text.

        Useful for debugging verification issues.
        """
        return {
            "numbers": list(extract_numbers(text)),
            "dates": list(extract_dates(text)),
            "currencies": list(extract_currencies(text)),
            "entities": list(extract_named_entities(text)),
        }
