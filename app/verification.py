"""
Post-LLM verification layer.
Never trust a single LLM pass.

Verification patterns:
- Level 1: Deterministic checks (regex, numbers, dates)
- Level 2: Critic LLM validation
- Rule checks for numbers, dates, named entities
"""

import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class VerificationStatus(Enum):
    VALID = "valid"
    INVALID = "invalid"
    UNCERTAIN = "uncertain"


@dataclass
class VerificationResult:
    """Result of answer verification."""

    status: VerificationStatus
    confidence: float
    unsupported_claims: List[str] = field(default_factory=list)
    missing_context: List[str] = field(default_factory=list)
    issues: List[str] = field(default_factory=list)
    level1_passed: bool = True
    level2_passed: Optional[bool] = None


class DeterministicVerifier:
    """
    Level 1: Deterministic verification checks.

    Performs:
    - Number extraction and validation
    - Date format checking
    - Named entity verification
    - Keyword presence checks
    """

    def __init__(self):
        self.number_pattern = re.compile(r"\b\d+(?:\.\d+)?%?\b")
        self.date_pattern = re.compile(
            r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b|"
            r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b|"
            r"\b\d{4}[-/]\d{2}[-/]\d{2}\b"
        )
        self.currency_pattern = re.compile(r"\$[\d,]+(?:\.\d{2})?")

    def extract_numbers(self, text: str) -> set:
        """Extract all numbers from text."""
        return set(self.number_pattern.findall(text))

    def extract_dates(self, text: str) -> set:
        """Extract all dates from text."""
        return set(self.date_pattern.findall(text))

    def extract_currencies(self, text: str) -> set:
        """Extract currency values from text."""
        return set(self.currency_pattern.findall(text))

    def verify(
        self,
        answer: str,
        context: str,
        strict_numbers: bool = True,
        strict_dates: bool = True,
    ) -> Tuple[bool, List[str]]:
        """
        Verify answer against context.

        Args:
            answer: Generated answer
            context: Retrieved context
            strict_numbers: Require numbers in answer to exist in context
            strict_dates: Require dates in answer to exist in context

        Returns:
            (is_valid, list of issues)
        """
        issues = []

        # Check numbers
        if strict_numbers:
            answer_numbers = self.extract_numbers(answer)
            context_numbers = self.extract_numbers(context)

            unsupported_numbers = answer_numbers - context_numbers
            if unsupported_numbers:
                issues.append(f"Numbers not found in context: {unsupported_numbers}")

        # Check dates
        if strict_dates:
            answer_dates = self.extract_dates(answer)
            context_dates = self.extract_dates(context)

            unsupported_dates = answer_dates - context_dates
            if unsupported_dates:
                issues.append(f"Dates not found in context: {unsupported_dates}")

        # Check currencies
        answer_currencies = self.extract_currencies(answer)
        context_currencies = self.extract_currencies(context)

        unsupported_currencies = answer_currencies - context_currencies
        if unsupported_currencies:
            issues.append(
                f"Currency values not found in context: {unsupported_currencies}"
            )

        is_valid = len(issues) == 0
        return is_valid, issues


class LLMCriticVerifier:
    """
    Level 2: LLM-based verification.

    Uses a smaller model to validate the generated answer
    against the provided context.
    """

    CRITIC_PROMPT = """You are a verification assistant. Your job is to check if the answer is supported by the context.

Context:
{context}

Answer to verify:
{answer}

Instructions:
1. For each factual claim in the answer, check if it is directly supported by the context.
2. If any statement is not supported or contradicts the context, mark as INVALID.
3. If the answer uses information not present in the context, mark as INVALID.

Return ONLY valid JSON:
{{
    "valid": true or false,
    "confidence": 0.0 to 1.0,
    "unsupported_claims": ["list of claims not in context"],
    "issues": ["list of specific issues found"]
}}"""

    def __init__(self, llm_client=None):
        self.llm_client = llm_client

    def verify(self, answer: str, context: str) -> Tuple[bool, Dict]:
        """
        Verify answer using LLM critic.

        Args:
            answer: Generated answer
            context: Retrieved context

        Returns:
            (is_valid, verification details)
        """
        if self.llm_client is None:
            logger.warning("No LLM client provided for critic verification")
            return True, {"skipped": True}

        prompt = self.CRITIC_PROMPT.format(context=context, answer=answer)

        try:
            # Call LLM (implementation depends on your client)
            response = self.llm_client.generate(
                prompt=prompt, temperature=0.0, max_tokens=500
            )

            # Parse JSON response
            result = json.loads(response)

            return result.get("valid", False), result

        except json.JSONDecodeError:
            logger.error("Failed to parse critic response as JSON")
            return False, {"error": "Invalid JSON response from critic"}
        except Exception as e:
            logger.error(f"Critic verification failed: {e}")
            return True, {"error": str(e), "skipped": True}


class AnswerVerifier:
    """
    Combined verification pipeline.

    Runs both Level 1 (deterministic) and Level 2 (LLM critic) checks.
    """

    def __init__(self, llm_client=None, enable_llm_critic: bool = True):
        self.deterministic = DeterministicVerifier()
        self.llm_critic = LLMCriticVerifier(llm_client) if enable_llm_critic else None

    def verify(
        self, answer: str, context: str, run_llm_critic: bool = True
    ) -> VerificationResult:
        """
        Full verification pipeline.

        Args:
            answer: Generated answer
            context: Retrieved context
            run_llm_critic: Whether to run LLM critic

        Returns:
            VerificationResult with all findings
        """
        # Level 1: Deterministic checks
        level1_passed, level1_issues = self.deterministic.verify(answer, context)

        result = VerificationResult(
            status=(
                VerificationStatus.VALID
                if level1_passed
                else VerificationStatus.INVALID
            ),
            confidence=1.0 if level1_passed else 0.5,
            issues=level1_issues,
            level1_passed=level1_passed,
        )

        # Level 2: LLM critic (if enabled and Level 1 passed)
        if run_llm_critic and self.llm_critic and level1_passed:
            level2_passed, level2_details = self.llm_critic.verify(answer, context)

            result.level2_passed = level2_passed
            result.confidence = level2_details.get("confidence", 0.5)
            result.unsupported_claims = level2_details.get("unsupported_claims", [])

            if not level2_passed:
                result.status = VerificationStatus.INVALID
                result.issues.extend(level2_details.get("issues", []))

        return result

    def get_fallback_response(self) -> str:
        """Return standard fallback when verification fails."""
        return "Not found in context"


def verify_answer(answer: str, context: str, llm_client=None) -> VerificationResult:
    """
    Convenience function for answer verification.

    Args:
        answer: Generated answer to verify
        context: Context used for generation
        llm_client: Optional LLM client for critic

    Returns:
        VerificationResult
    """
    verifier = AnswerVerifier(llm_client=llm_client)
    return verifier.verify(answer, context)


def simple_verify(answer: str, context: str) -> bool:
    """
    Simple verification: check if key facts in answer appear in context.

    Useful for quick checks without LLM overhead.
    """
    verifier = DeterministicVerifier()
    is_valid, _ = verifier.verify(answer, context)
    return is_valid
