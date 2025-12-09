"""
Unit economics calculator for RAG operations.

Tracks cost per query, token usage, and ROI metrics.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class CostBreakdown:
    """Cost breakdown for a single operation."""

    embedding_cost: float = 0.0
    retrieval_cost: float = 0.0
    reranking_cost: float = 0.0
    llm_cost: float = 0.0
    storage_cost: float = 0.0

    @property
    def total(self) -> float:
        """Total cost."""
        return (
            self.embedding_cost
            + self.retrieval_cost
            + self.reranking_cost
            + self.llm_cost
            + self.storage_cost
        )


class UnitEconomicsCalculator:
    """
    Calculate unit economics for RAG operations.

    Tracks:
    - Cost per query
    - Token usage
    - Cache hit rate impact
    - Model routing savings
    """

    # Default pricing (update with actual rates)
    PRICING = {
        "openai_gpt4": 0.03 / 1000,  # $0.03 per 1K tokens
        "openai_gpt4_mini": 0.001 / 1000,  # $0.001 per 1K tokens
        "embedding": 0.0001 / 1000,  # $0.0001 per 1K tokens
        "reranking": 0.0005 / 1000,  # Estimated
    }

    def __init__(self):
        """Initialize calculator."""
        self.total_queries = 0
        self.total_cost = 0.0
        self.total_tokens = 0
        self.cache_hits = 0

    def calculate_query_cost(
        self, tokens_used: int, model: str = "gpt4_mini", cache_hit: bool = False
    ) -> CostBreakdown:
        """
        Calculate cost for a single query.

        Args:
            tokens_used: Number of tokens used
            model: Model name
            cache_hit: Whether this was a cache hit

        Returns:
            Cost breakdown
        """
        if cache_hit:
            return CostBreakdown()  # No cost for cache hits

        model_key = f"openai_{model}"
        llm_cost = tokens_used * self.PRICING.get(model_key, 0.0)

        breakdown = CostBreakdown(llm_cost=llm_cost)

        self.total_queries += 1
        self.total_cost += breakdown.total
        self.total_tokens += tokens_used

        if cache_hit:
            self.cache_hits += 1

        return breakdown

    def get_summary(self) -> Dict:
        """Get cost summary."""
        cache_hit_rate = (
            self.cache_hits / self.total_queries if self.total_queries > 0 else 0.0
        )

        avg_cost_per_query = (
            self.total_cost / self.total_queries if self.total_queries > 0 else 0.0
        )

        return {
            "total_queries": self.total_queries,
            "total_cost": self.total_cost,
            "total_tokens": self.total_tokens,
            "cache_hit_rate": cache_hit_rate,
            "avg_cost_per_query": avg_cost_per_query,
            "estimated_monthly_cost": avg_cost_per_query * 100000,  # 100K queries/month
        }


# Global calculator
_calculator: Optional[UnitEconomicsCalculator] = None


def get_cost_calculator() -> UnitEconomicsCalculator:
    """Get global cost calculator."""
    global _calculator
    if _calculator is None:
        _calculator = UnitEconomicsCalculator()
    return _calculator
