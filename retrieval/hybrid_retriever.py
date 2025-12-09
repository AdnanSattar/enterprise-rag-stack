"""
Hybrid retrieval combining vector + lexical + metadata.

Retrieval is the optical lens for your LLM.

Best practices:
- Hybrid retrieval beats pure vector for keyword-heavy queries
- Score fusion: score = w_vec * s_vec + w_bm25 * s_bm25
- Dynamic TopK based on query complexity
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .lexical_retriever import compute_bm25_score
from .score_fusion import QueryClassifier
from .vector_retriever import VectorStore

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """A single retrieval result with scores."""

    id: str
    doc_id: str
    text: str
    metadata: Dict
    vector_score: float
    lexical_score: float
    combined_score: float


class HybridRetriever:
    """
    Hybrid retrieval combining vector + lexical + metadata.

    Score fusion: score = w_vec * s_vec + w_bm25 * s_bm25

    Usage:
        retriever = HybridRetriever()
        results = retriever.retrieve(
            "What are the payment terms?",
            top_k=5,
            filters={"tenant_id": "acme"}
        )
    """

    def __init__(
        self,
        vector_store: VectorStore = None,
        vector_weight: float = 0.7,
        lexical_weight: float = 0.3,
        use_query_classification: bool = True,
    ):
        """
        Args:
            vector_store: VectorStore instance
            vector_weight: Weight for vector similarity
            lexical_weight: Weight for BM25 score
            use_query_classification: Adjust weights based on query type
        """
        self.vector_store = vector_store or VectorStore()
        self.vector_weight = vector_weight
        self.lexical_weight = lexical_weight
        self.use_query_classification = use_query_classification
        self._query_classifier = QueryClassifier() if use_query_classification else None

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict] = None,
        use_bm25: bool = True,
    ) -> List[RetrievalResult]:
        """
        Hybrid retrieval with score fusion.

        Args:
            query: Search query
            top_k: Number of results
            filters: Metadata filters
            use_bm25: Use BM25 scoring (vs simple overlap)

        Returns:
            Ranked list of RetrievalResult
        """
        # Adjust weights based on query type
        vec_weight = self.vector_weight
        lex_weight = self.lexical_weight

        if self._query_classifier:
            query_type, weights = self._query_classifier.classify(query)
            vec_weight = weights["vector_weight"]
            lex_weight = weights["lexical_weight"]
            logger.debug(
                f"Query type: {query_type}, weights: vec={vec_weight}, lex={lex_weight}"
            )

        # 1. Vector search for semantic candidates
        # Get more candidates than needed for fusion
        vector_results = self.vector_store.query(
            query_text=query,
            n_results=top_k * 2,
            where=filters,
        )

        if not vector_results["ids"][0]:
            return []

        # 2. Score fusion
        results = []
        docs = vector_results["documents"][0]
        metas = vector_results["metadatas"][0]
        ids = vector_results["ids"][0]
        distances = vector_results["distances"][0]

        for i, (doc_id, text, meta, dist) in enumerate(
            zip(ids, docs, metas, distances)
        ):
            # Convert distance to similarity score
            # Chroma returns L2 distance for cosine space
            vector_score = 1 - (dist / 2)  # Approximate conversion

            # Lexical score
            if use_bm25:
                lex_score = compute_bm25_score(query, text)
            else:
                lex_score = self._lexical_overlap(query, text)

            # Normalize lexical score to [0, 1] range
            lex_score = min(lex_score / 5.0, 1.0)

            # Weighted fusion
            combined = vec_weight * vector_score + lex_weight * lex_score

            results.append(
                RetrievalResult(
                    id=doc_id,
                    doc_id=meta.get("doc_id", doc_id.split("#")[0]),
                    text=text,
                    metadata=meta,
                    vector_score=vector_score,
                    lexical_score=lex_score,
                    combined_score=combined,
                )
            )

        # 3. Sort by combined score
        results.sort(key=lambda x: x.combined_score, reverse=True)

        return results[:top_k]

    def _lexical_overlap(self, query: str, text: str) -> float:
        """Simple lexical overlap score."""
        q_terms = set(query.lower().split())
        t_terms = set(text.lower().split())

        if not q_terms:
            return 0.0

        overlap = len(q_terms & t_terms)
        return overlap / (len(q_terms) + 1e-6)

    def retrieve_with_query_expansion(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict] = None,
    ) -> List[RetrievalResult]:
        """
        Retrieval with query expansion.

        Canonicalize user query before embedding to improve recall.
        """
        expanded_query = self._expand_query(query)
        return self.retrieve(expanded_query, top_k, filters)

    def _expand_query(self, query: str) -> str:
        """Simple query expansion/canonicalization."""
        # Remove filler words
        fillers = {"please", "can", "you", "tell", "me", "about", "what", "is", "the"}
        words = query.split()
        expanded = [w for w in words if w.lower() not in fillers]

        # If query is too short after expansion, use original
        if len(expanded) < 2:
            return query

        return " ".join(expanded)


# Convenience functions
_retriever: Optional[HybridRetriever] = None


def get_retriever(
    vector_weight: float = 0.7,
    lexical_weight: float = 0.3,
) -> HybridRetriever:
    """Get or create global retriever instance."""
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever(
            vector_weight=vector_weight,
            lexical_weight=lexical_weight,
        )
    return _retriever


def retrieve(
    query: str,
    top_k: int = 10,
    filters: Optional[Dict] = None,
) -> List[RetrievalResult]:
    """Convenience function for retrieval."""
    return get_retriever().retrieve(query, top_k, filters)
