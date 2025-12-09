"""
Verification Layer Module.

Never trust a single LLM pass.

This module implements two-level verification:
- Level 1: Deterministic checks (regex on numbers, dates, allowed values)
- Level 2: Critic LLM that validates answer against context

Usage:
    from verification import AnswerVerifier, verify_answer

    verifier = AnswerVerifier()
    result = verifier.verify(answer, context)
"""

from .critic_llm import LLMCriticVerifier
from .deterministic_checks import (
    DeterministicVerifier,
    extract_currencies,
    extract_dates,
    extract_numbers,
)
from .verification_pipeline import (
    AnswerVerifier,
    VerificationResult,
    VerificationStatus,
    simple_verify,
    verify_answer,
)

__all__ = [
    "DeterministicVerifier",
    "extract_numbers",
    "extract_dates",
    "extract_currencies",
    "LLMCriticVerifier",
    "AnswerVerifier",
    "VerificationResult",
    "VerificationStatus",
    "verify_answer",
    "simple_verify",
]
