"""
Reranking Pipeline Module.

Reranking is cost-effective accuracy.

Key points:
- Cross-encoder reranker for top candidates (20-50)
- Merge semantic and lexical signals
- Deduplicate overlapping chunks
- Only run on limited candidates (expensive per query)

Usage:
    from reranking import CrossEncoderReranker, rerank_results

    reranker = CrossEncoderReranker()
    results = reranker.rerank(query, candidates, top_k=5)
"""

from .batch_reranking import BatchReranker, deduplicate_chunks
from .cross_encoder_reranker import (
    CrossEncoderReranker,
    LightweightReranker,
    RerankResult,
    get_reranker,
    rerank_results,
)

__all__ = [
    "CrossEncoderReranker",
    "LightweightReranker",
    "RerankResult",
    "get_reranker",
    "rerank_results",
    "BatchReranker",
    "deduplicate_chunks",
]
