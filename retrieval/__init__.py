"""
Hybrid Retrieval Module.

Retrieval is the optical lens for your LLM.

This module implements:
- Vector retrieval (semantic similarity)
- Lexical retrieval (BM25)
- Hybrid retrieval (score fusion)
- Metadata filtering

Best practices:
- Hybrid retrieval: vector + BM25 + metadata filters
- Score fusion: score = w_vec * s_vec + w_bm25 * s_bm25
- Query classification for dynamic TopK

Usage:
    from retrieval import HybridRetriever, VectorStore

    retriever = HybridRetriever()
    results = retriever.retrieve("What are the payment terms?", top_k=5)
"""

from .hybrid_retriever import HybridRetriever, RetrievalResult
from .lexical_retriever import BM25Retriever, compute_bm25_score
from .score_fusion import FusionMethod, fuse_scores, normalize_scores
from .vector_retriever import VectorRetriever, VectorStore

__all__ = [
    "VectorStore",
    "VectorRetriever",
    "BM25Retriever",
    "compute_bm25_score",
    "HybridRetriever",
    "RetrievalResult",
    "normalize_scores",
    "fuse_scores",
    "FusionMethod",
]
