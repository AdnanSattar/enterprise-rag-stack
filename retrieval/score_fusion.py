"""
Score fusion utilities for hybrid retrieval.

Combines scores from multiple retrieval methods:
- Vector similarity (semantic)
- BM25 (lexical)
- Metadata-based boosts
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class FusionMethod(Enum):
    """Score fusion methods."""

    WEIGHTED_SUM = "weighted_sum"
    RECIPROCAL_RANK = "reciprocal_rank"
    RELATIVE_SCORE = "relative_score"


@dataclass
class FusedResult:
    """Result after score fusion."""

    id: str
    vector_score: float
    lexical_score: float
    fused_score: float
    rank: int


def normalize_scores(
    scores: List[float],
    method: str = "minmax",
) -> List[float]:
    """
    Normalize scores to [0, 1] range.

    Args:
        scores: Raw scores
        method: Normalization method (minmax, zscore)

    Returns:
        Normalized scores
    """
    if not scores:
        return []

    scores = np.array(scores)

    if method == "minmax":
        min_s = scores.min()
        max_s = scores.max()
        if max_s - min_s == 0:
            return [1.0] * len(scores)
        return list((scores - min_s) / (max_s - min_s))

    elif method == "zscore":
        mean_s = scores.mean()
        std_s = scores.std()
        if std_s == 0:
            return [0.5] * len(scores)
        # Z-score then sigmoid to [0, 1]
        z = (scores - mean_s) / std_s
        return list(1 / (1 + np.exp(-z)))

    else:
        raise ValueError(f"Unknown normalization method: {method}")


def fuse_scores(
    vector_results: Dict[str, float],
    lexical_results: Dict[str, float],
    vector_weight: float = 0.7,
    lexical_weight: float = 0.3,
    method: FusionMethod = FusionMethod.WEIGHTED_SUM,
) -> List[FusedResult]:
    """
    Fuse scores from vector and lexical retrieval.

    Formula (weighted sum): score = w_vec * s_vec + w_bm25 * s_bm25

    Args:
        vector_results: {doc_id: score} from vector search
        lexical_results: {doc_id: score} from lexical search
        vector_weight: Weight for vector scores
        lexical_weight: Weight for lexical scores
        method: Fusion method

    Returns:
        Sorted list of FusedResult
    """
    # Get all document IDs
    all_ids = set(vector_results.keys()) | set(lexical_results.keys())

    if not all_ids:
        return []

    # Normalize scores
    if vector_results:
        vec_scores = list(vector_results.values())
        vec_norm = normalize_scores(vec_scores)
        vec_normalized = dict(zip(vector_results.keys(), vec_norm))
    else:
        vec_normalized = {}

    if lexical_results:
        lex_scores = list(lexical_results.values())
        lex_norm = normalize_scores(lex_scores)
        lex_normalized = dict(zip(lexical_results.keys(), lex_norm))
    else:
        lex_normalized = {}

    # Fuse scores
    results = []

    if method == FusionMethod.WEIGHTED_SUM:
        for doc_id in all_ids:
            vec_score = vec_normalized.get(doc_id, 0.0)
            lex_score = lex_normalized.get(doc_id, 0.0)

            fused = vector_weight * vec_score + lexical_weight * lex_score

            results.append(
                FusedResult(
                    id=doc_id,
                    vector_score=vector_results.get(doc_id, 0.0),
                    lexical_score=lexical_results.get(doc_id, 0.0),
                    fused_score=fused,
                    rank=0,  # Will be set after sorting
                )
            )

    elif method == FusionMethod.RECIPROCAL_RANK:
        # RRF: score = sum(1 / (k + rank))
        k = 60  # Constant, typical value

        # Get ranks from each system
        vec_ranks = _get_ranks(vector_results)
        lex_ranks = _get_ranks(lexical_results)

        for doc_id in all_ids:
            vec_rank = vec_ranks.get(doc_id, 1000)
            lex_rank = lex_ranks.get(doc_id, 1000)

            rrf = vector_weight * (1 / (k + vec_rank)) + lexical_weight * (
                1 / (k + lex_rank)
            )

            results.append(
                FusedResult(
                    id=doc_id,
                    vector_score=vector_results.get(doc_id, 0.0),
                    lexical_score=lexical_results.get(doc_id, 0.0),
                    fused_score=rrf,
                    rank=0,
                )
            )

    # Sort by fused score
    results.sort(key=lambda x: x.fused_score, reverse=True)

    # Assign ranks
    for i, result in enumerate(results):
        result.rank = i + 1

    return results


def _get_ranks(scores: Dict[str, float]) -> Dict[str, int]:
    """Convert scores to ranks."""
    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return {doc_id: rank + 1 for rank, (doc_id, _) in enumerate(sorted_items)}


def apply_metadata_boost(
    results: List[FusedResult],
    boost_field: str,
    boost_values: Dict[str, float],
    metadata_lookup: Dict[str, Dict],
) -> List[FusedResult]:
    """
    Apply metadata-based score boosts.

    Example: Boost recent documents, boost specific sources

    Args:
        results: Fused results
        boost_field: Metadata field to use for boosting
        boost_values: {field_value: boost_multiplier}
        metadata_lookup: {doc_id: metadata_dict}

    Returns:
        Re-sorted results with boosts applied
    """
    for result in results:
        metadata = metadata_lookup.get(result.id, {})
        field_value = metadata.get(boost_field)

        if field_value and field_value in boost_values:
            boost = boost_values[field_value]
            result.fused_score *= boost

    # Re-sort and re-rank
    results.sort(key=lambda x: x.fused_score, reverse=True)
    for i, result in enumerate(results):
        result.rank = i + 1

    return results


class QueryClassifier:
    """
    Classify queries to adjust retrieval strategy.

    Query types:
    - factual: Specific facts, use more lexical weight
    - exploratory: Broad topics, use more semantic weight
    - navigational: Looking for specific documents
    """

    def classify(self, query: str) -> Tuple[str, Dict[str, float]]:
        """
        Classify query and return adjusted weights.

        Args:
            query: User query

        Returns:
            (query_type, {"vector_weight": x, "lexical_weight": y})
        """
        query_lower = query.lower()

        # Factual indicators
        factual_keywords = [
            "what is",
            "how much",
            "when",
            "where",
            "price",
            "cost",
            "date",
            "number",
            "amount",
        ]

        # Exploratory indicators
        exploratory_keywords = [
            "explain",
            "describe",
            "overview",
            "summary",
            "tell me about",
            "what are",
            "how does",
        ]

        # Check for factual
        if any(kw in query_lower for kw in factual_keywords):
            return "factual", {"vector_weight": 0.5, "lexical_weight": 0.5}

        # Check for exploratory
        if any(kw in query_lower for kw in exploratory_keywords):
            return "exploratory", {"vector_weight": 0.8, "lexical_weight": 0.2}

        # Check for codes/IDs (navigational)
        import re

        if re.search(r"[A-Z]{2,}\d+|\d{3,}", query):
            return "navigational", {"vector_weight": 0.3, "lexical_weight": 0.7}

        # Default balanced
        return "general", {"vector_weight": 0.7, "lexical_weight": 0.3}
