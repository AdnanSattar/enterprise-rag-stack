"""
Lexical retrieval using BM25.

Pure vector retrieval fails for keyword-heavy queries
(e.g., SKU numbers, technical codes). BM25 captures exact
term matches that semantic search misses.
"""

import logging
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class BM25Result:
    """Result from BM25 search."""

    id: str
    text: str
    score: float
    term_matches: Dict[str, int]


def compute_bm25_score(
    query: str,
    text: str,
    k1: float = 1.5,
    b: float = 0.75,
    avg_doc_len: float = 500,
) -> float:
    """
    BM25 scoring for lexical retrieval.

    Args:
        query: Query string
        text: Document text
        k1: Term frequency saturation parameter (1.2-2.0 typical)
        b: Length normalization parameter (0.75 typical)
        avg_doc_len: Average document length in corpus

    Returns:
        BM25 score
    """
    q_terms = query.lower().split()
    doc_terms = text.lower().split()
    doc_len = len(doc_terms)

    if doc_len == 0:
        return 0.0

    # Term frequency in document
    tf = defaultdict(int)
    for term in doc_terms:
        tf[term] += 1

    score = 0.0
    for term in q_terms:
        if term in tf:
            freq = tf[term]
            # BM25 formula
            numerator = freq * (k1 + 1)
            denominator = freq + k1 * (1 - b + b * (doc_len / avg_doc_len))
            score += numerator / denominator

    return score


def lexical_overlap_score(query: str, text: str) -> float:
    """
    Simple lexical overlap score.

    Faster than BM25, useful for quick filtering.

    Args:
        query: Query string
        text: Document text

    Returns:
        Overlap score (0-1)
    """
    q_terms = set(query.lower().split())
    t_terms = set(text.lower().split())

    if not q_terms:
        return 0.0

    overlap = len(q_terms & t_terms)
    return overlap / len(q_terms)


class BM25Index:
    """
    In-memory BM25 index.

    For production with large corpora, use Elasticsearch or OpenSearch.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b

        self._documents: Dict[str, str] = {}
        self._doc_lengths: Dict[str, int] = {}
        self._avg_doc_len: float = 0
        self._term_doc_freq: Dict[str, int] = defaultdict(int)
        self._inverted_index: Dict[str, Dict[str, int]] = defaultdict(dict)

    def add_documents(
        self,
        ids: List[str],
        texts: List[str],
    ) -> None:
        """Add documents to the index."""
        for doc_id, text in zip(ids, texts):
            self._documents[doc_id] = text
            terms = text.lower().split()
            self._doc_lengths[doc_id] = len(terms)

            # Update term frequencies
            seen_terms = set()
            for term in terms:
                if term not in seen_terms:
                    self._term_doc_freq[term] += 1
                    seen_terms.add(term)

                # Update inverted index
                if doc_id not in self._inverted_index[term]:
                    self._inverted_index[term][doc_id] = 0
                self._inverted_index[term][doc_id] += 1

        # Update average document length
        if self._doc_lengths:
            self._avg_doc_len = sum(self._doc_lengths.values()) / len(self._doc_lengths)

    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[BM25Result]:
        """
        Search the index.

        Args:
            query: Search query
            top_k: Number of results

        Returns:
            List of BM25Result sorted by score
        """
        q_terms = query.lower().split()
        n_docs = len(self._documents)

        if n_docs == 0:
            return []

        scores: Dict[str, float] = defaultdict(float)
        term_matches: Dict[str, Dict[str, int]] = defaultdict(dict)

        for term in q_terms:
            if term not in self._inverted_index:
                continue

            # IDF component
            df = self._term_doc_freq.get(term, 0)
            idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1)

            # Score each document containing this term
            for doc_id, tf in self._inverted_index[term].items():
                doc_len = self._doc_lengths[doc_id]

                # BM25 TF component
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (
                    1 - self.b + self.b * (doc_len / self._avg_doc_len)
                )

                scores[doc_id] += idf * (numerator / denominator)
                term_matches[doc_id][term] = tf

        # Sort by score
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        results = []
        for doc_id, score in sorted_docs:
            results.append(
                BM25Result(
                    id=doc_id,
                    text=self._documents[doc_id],
                    score=score,
                    term_matches=dict(term_matches[doc_id]),
                )
            )

        return results

    def get_stats(self) -> Dict:
        """Get index statistics."""
        return {
            "total_documents": len(self._documents),
            "unique_terms": len(self._inverted_index),
            "avg_doc_length": self._avg_doc_len,
        }


class BM25Retriever:
    """
    BM25-based lexical retriever.

    Usage:
        retriever = BM25Retriever()
        retriever.index_documents(ids, texts)
        results = retriever.retrieve("search query", top_k=10)
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.index = BM25Index(k1=k1, b=b)

    def index_documents(
        self,
        ids: List[str],
        texts: List[str],
    ) -> None:
        """Add documents to the index."""
        self.index.add_documents(ids, texts)
        logger.info(f"Indexed {len(ids)} documents for BM25")

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[BM25Result]:
        """
        Retrieve documents using BM25.

        Args:
            query: Search query
            top_k: Number of results

        Returns:
            List of BM25Result
        """
        return self.index.search(query, top_k)
