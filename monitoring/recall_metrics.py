"""
Recall metrics computation for RAG evaluation.

Measures retrieval quality:
- Recall@K: Fraction of relevant documents retrieved
- Precision@K: Fraction of retrieved documents that are relevant
- MRR: Mean Reciprocal Rank
- NDCG: Normalized Discounted Cumulative Gain
"""

from typing import List, Set


def recall_at_k(
    retrieved_ids: List[str], relevant_ids: List[str], k: int = None
) -> float:
    """
    Compute Recall@K.

    What fraction of relevant documents were retrieved?

    Args:
        retrieved_ids: List of retrieved document IDs (ordered by relevance)
        relevant_ids: List of relevant document IDs
        k: Cutoff rank (if None, uses all retrieved)

    Returns:
        Recall score between 0.0 and 1.0
    """
    if not relevant_ids:
        return 1.0

    if k:
        retrieved_ids = retrieved_ids[:k]

    retrieved_set = set(retrieved_ids)
    relevant_set = set(relevant_ids)

    found = len(retrieved_set & relevant_set)
    return found / len(relevant_set)


def precision_at_k(
    retrieved_ids: List[str], relevant_ids: List[str], k: int = None
) -> float:
    """
    Compute Precision@K.

    What fraction of retrieved documents were relevant?

    Args:
        retrieved_ids: List of retrieved document IDs
        relevant_ids: List of relevant document IDs
        k: Cutoff rank

    Returns:
        Precision score between 0.0 and 1.0
    """
    if k:
        retrieved_ids = retrieved_ids[:k]

    if not retrieved_ids:
        return 0.0

    retrieved_set = set(retrieved_ids)
    relevant_set = set(relevant_ids)

    relevant = len(retrieved_set & relevant_set)
    return relevant / len(retrieved_set)


def mean_reciprocal_rank(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
    """
    Compute Mean Reciprocal Rank (MRR).

    Where does the first relevant document appear?

    Args:
        retrieved_ids: List of retrieved document IDs
        relevant_ids: List of relevant document IDs

    Returns:
        MRR score (1/rank of first relevant, or 0.0 if none found)
    """
    relevant_set = set(relevant_ids)

    for i, doc_id in enumerate(retrieved_ids):
        if doc_id in relevant_set:
            return 1.0 / (i + 1)

    return 0.0


def ndcg_at_k(
    retrieved_ids: List[str], relevant_ids: List[str], k: int = None
) -> float:
    """
    Compute Normalized Discounted Cumulative Gain@K.

    Measures ranking quality with position discounting.

    Args:
        retrieved_ids: List of retrieved document IDs
        relevant_ids: List of relevant document IDs
        k: Cutoff rank

    Returns:
        NDCG score between 0.0 and 1.0
    """
    import math

    if k:
        retrieved_ids = retrieved_ids[:k]

    relevant_set = set(relevant_ids)

    # Compute DCG
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_ids):
        if doc_id in relevant_set:
            dcg += 1.0 / math.log2(i + 2)  # +2 because log2(1) = 0

    # Compute ideal DCG
    ideal_dcg = sum(
        1.0 / math.log2(i + 2)
        for i in range(min(len(relevant_ids), len(retrieved_ids)))
    )

    if ideal_dcg == 0:
        return 0.0

    return dcg / ideal_dcg
