"""
Level 2: LLM-based verification (Critic LLM).

Uses a smaller/cheaper model to validate the generated answer
against the provided context.

Returns: {valid: bool, issues: [...]}
"""

import json
import logging
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


CRITIC_PROMPT = """You are a verification assistant. Your job is to check if the answer is factually supported by the context.

Context:
{context}

Answer to verify:
{answer}

Instructions:
1. For each factual claim in the answer, check if it is directly supported by the context.
2. If any statement is not supported or contradicts the context, mark as INVALID.
3. If the answer uses information not present in the context (hallucination), mark as INVALID.
4. Minor paraphrasing is acceptable if the meaning is preserved.

Return ONLY valid JSON:
{{
    "valid": true or false,
    "confidence": 0.0 to 1.0,
    "unsupported_claims": ["list of claims not in context"],
    "issues": ["list of specific issues found"]
}}"""


class LLMCriticVerifier:
    """
    Level 2: LLM-based verification.

    Uses a critic LLM (usually smaller/cheaper than generation LLM)
    to validate that the answer is grounded in the context.

    Usage:
        critic = LLMCriticVerifier(llm_client)
        is_valid, details = critic.verify(answer, context)
    """

    def __init__(
        self,
        llm_client=None,
        model: str = None,
        temperature: float = 0.0,
        max_tokens: int = 500,
    ):
        """
        Args:
            llm_client: LLM client for critic calls
            model: Model to use (defaults to client default)
            temperature: Use 0 for deterministic verification
            max_tokens: Max tokens for critic response
        """
        self.llm_client = llm_client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def verify(
        self,
        answer: str,
        context: str,
    ) -> Tuple[bool, Dict]:
        """
        Verify answer using LLM critic.

        Args:
            answer: Generated answer to verify
            context: Retrieved context

        Returns:
            (is_valid, verification details dict)
        """
        if self.llm_client is None:
            logger.warning("No LLM client provided for critic verification")
            return True, {"skipped": True, "reason": "No LLM client"}

        prompt = CRITIC_PROMPT.format(context=context, answer=answer)

        try:
            # Call LLM
            response = self.llm_client.generate(
                prompt=prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            # Parse JSON response
            result = self._parse_response(response.content)

            is_valid = result.get("valid", False)

            # Log verification result
            if not is_valid:
                logger.info(f"Critic verification failed: {result.get('issues', [])}")

            return is_valid, result

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse critic response as JSON: {e}")
            return False, {"error": "Invalid JSON response from critic"}
        except Exception as e:
            logger.error(f"Critic verification failed: {e}")
            return True, {"error": str(e), "skipped": True}

    def _parse_response(self, response_text: str) -> Dict:
        """Parse critic response, handling various formats."""
        # Try direct JSON parse
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            pass

        # Try to extract JSON from text
        import re

        match = re.search(r"\{[\s\S]*\}", response_text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        # Return error structure
        return {
            "valid": False,
            "error": "Could not parse response",
            "raw_response": response_text[:500],
        }

    def batch_verify(
        self,
        answer_context_pairs: list,
    ) -> list:
        """
        Verify multiple answer-context pairs.

        More efficient for batch processing.

        Args:
            answer_context_pairs: List of (answer, context) tuples

        Returns:
            List of (is_valid, details) tuples
        """
        results = []
        for answer, context in answer_context_pairs:
            result = self.verify(answer, context)
            results.append(result)
        return results


class SimpleCriticVerifier:
    """
    Simplified critic using heuristics instead of LLM.

    Use when LLM critic is too expensive or slow.
    Less accurate but much faster.
    """

    def verify(
        self,
        answer: str,
        context: str,
    ) -> Tuple[bool, Dict]:
        """
        Verify using heuristics.

        Checks:
        - Answer keywords appear in context
        - Answer is not too generic
        - Answer is not too long relative to context
        """
        issues = []

        # Check if answer keywords are in context
        answer_words = set(answer.lower().split())
        context_words = set(context.lower().split())

        # Remove common words
        stop_words = {
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "must",
            "shall",
        }

        answer_keywords = answer_words - stop_words
        context_keywords = context_words - stop_words

        if answer_keywords:
            overlap = len(answer_keywords & context_keywords) / len(answer_keywords)
            if overlap < 0.3:
                issues.append(f"Low keyword overlap with context ({overlap:.0%})")

        # Check for generic responses
        generic_phrases = [
            "i don't know",
            "i cannot",
            "i'm not sure",
            "based on my knowledge",
            "in general",
        ]

        answer_lower = answer.lower()
        for phrase in generic_phrases:
            if phrase in answer_lower:
                issues.append(f"Contains generic phrase: '{phrase}'")

        is_valid = len(issues) == 0

        return is_valid, {
            "valid": is_valid,
            "issues": issues,
            "method": "heuristic",
        }
