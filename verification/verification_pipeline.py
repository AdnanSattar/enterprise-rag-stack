"""
Combined verification pipeline.

Runs both Level 1 (deterministic) and Level 2 (LLM critic) checks.

Two-level verification:
- Level 1: Fast deterministic checks (always run)
- Level 2: LLM critic (run if Level 1 passes)
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from .critic_llm import LLMCriticVerifier
from .deterministic_checks import DeterministicVerifier

logger = logging.getLogger(__name__)


class VerificationStatus(Enum):
    """Verification status."""

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


class AnswerVerifier:
    """
    Combined verification pipeline.

    Runs both Level 1 (deterministic) and Level 2 (LLM critic) checks.

    Usage:
        verifier = AnswerVerifier()
        result = verifier.verify(answer, context)

        if result.status == VerificationStatus.INVALID:
            # Use fallback response
            answer = verifier.get_fallback_response()
    """

    def __init__(
        self,
        llm_client=None,
        enable_llm_critic: bool = True,
        strict_numbers: bool = True,
        strict_dates: bool = True,
    ):
        """
        Args:
            llm_client: LLM client for critic verification
            enable_llm_critic: Enable Level 2 verification
            strict_numbers: Check numbers in Level 1
            strict_dates: Check dates in Level 1
        """
        self.deterministic = DeterministicVerifier(
            strict_numbers=strict_numbers,
            strict_dates=strict_dates,
        )
        self.llm_critic = (
            LLMCriticVerifier(llm_client) if enable_llm_critic and llm_client else None
        )

    def verify(
        self,
        answer: str,
        context: str,
        run_llm_critic: bool = True,
    ) -> VerificationResult:
        """
        Full verification pipeline.

        Args:
            answer: Generated answer
            context: Retrieved context
            run_llm_critic: Whether to run LLM critic (Level 2)

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


def verify_answer(
    answer: str,
    context: str,
    llm_client=None,
) -> VerificationResult:
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

    Args:
        answer: Generated answer
        context: Retrieved context

    Returns:
        True if answer appears grounded in context
    """
    verifier = DeterministicVerifier()
    is_valid, _ = verifier.verify(answer, context)
    return is_valid
