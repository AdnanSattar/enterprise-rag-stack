"""
Content deduplication utilities.

Prevents duplicate content from polluting your index.
Uses content hashing for exact dedup and optional
semantic similarity for near-duplicate detection.
"""

import hashlib
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)


def compute_content_hash(text: str, algorithm: str = "md5") -> str:
    """
    Compute content hash for deduplication.

    Normalizes whitespace before hashing to catch
    minor formatting differences.

    Args:
        text: Text content to hash
        algorithm: Hash algorithm (md5, sha256)

    Returns:
        Hex digest of content hash

    Example:
        >>> hash1 = compute_content_hash("Hello  World")
        >>> hash2 = compute_content_hash("Hello World")
        >>> hash1 == hash2  # Same after normalization
        True
    """
    # Normalize whitespace before hashing
    normalized = " ".join(text.split())

    if algorithm == "sha256":
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return hashlib.md5(normalized.encode("utf-8")).hexdigest()


@dataclass
class DedupeResult:
    """Result of deduplication check."""

    is_duplicate: bool
    existing_doc_id: Optional[str] = None
    similarity_score: Optional[float] = None


class ContentDeduplicator:
    """
    Content deduplication with exact and near-duplicate detection.

    Usage:
        deduplicator = ContentDeduplicator()

        # Check if content is duplicate
        result = deduplicator.check(content_hash, text)
        if not result.is_duplicate:
            deduplicator.add(content_hash, doc_id, text)
    """

    def __init__(
        self,
        enable_semantic_dedup: bool = False,
        similarity_threshold: float = 0.95,
    ):
        """
        Args:
            enable_semantic_dedup: Enable near-duplicate detection
            similarity_threshold: Threshold for semantic similarity (0-1)
        """
        self._seen_hashes: Set[str] = set()
        self._hash_to_doc: Dict[str, str] = {}
        self._embeddings: Dict[str, List[float]] = {}
        self.enable_semantic_dedup = enable_semantic_dedup
        self.similarity_threshold = similarity_threshold

    def check(self, content_hash: str, text: str = None) -> DedupeResult:
        """
        Check if content is a duplicate.

        Args:
            content_hash: Hash of the content
            text: Original text (for semantic dedup)

        Returns:
            DedupeResult indicating if duplicate
        """
        # Exact hash match
        if content_hash in self._seen_hashes:
            return DedupeResult(
                is_duplicate=True,
                existing_doc_id=self._hash_to_doc.get(content_hash),
            )

        # Semantic similarity check (expensive, use sparingly)
        if self.enable_semantic_dedup and text and self._embeddings:
            similarity, existing_id = self._check_semantic_similarity(text)
            if similarity >= self.similarity_threshold:
                return DedupeResult(
                    is_duplicate=True,
                    existing_doc_id=existing_id,
                    similarity_score=similarity,
                )

        return DedupeResult(is_duplicate=False)

    def add(
        self,
        content_hash: str,
        doc_id: str,
        text: str = None,
        embedding: List[float] = None,
    ) -> None:
        """
        Add a document to the deduplication index.

        Args:
            content_hash: Hash of the content
            doc_id: Document identifier
            text: Original text (for logging)
            embedding: Pre-computed embedding (for semantic dedup)
        """
        self._seen_hashes.add(content_hash)
        self._hash_to_doc[content_hash] = doc_id

        if self.enable_semantic_dedup and embedding:
            self._embeddings[doc_id] = embedding

        logger.debug(f"Added doc {doc_id} to dedup index")

    def remove(self, content_hash: str) -> bool:
        """
        Remove a document from the deduplication index.

        Args:
            content_hash: Hash of content to remove

        Returns:
            True if removed, False if not found
        """
        if content_hash not in self._seen_hashes:
            return False

        self._seen_hashes.discard(content_hash)
        doc_id = self._hash_to_doc.pop(content_hash, None)

        if doc_id and doc_id in self._embeddings:
            del self._embeddings[doc_id]

        return True

    def _check_semantic_similarity(self, text: str) -> tuple[float, Optional[str]]:
        """
        Check semantic similarity against existing documents.

        Returns:
            (max_similarity, matching_doc_id)
        """
        try:
            import numpy as np
            from embeddings import get_embedding_service

            service = get_embedding_service()
            result = service.embed_batch([text])
            query_vec = result.vectors[0]

            max_sim = 0.0
            max_doc = None

            for doc_id, emb in self._embeddings.items():
                sim = np.dot(query_vec, emb)
                if sim > max_sim:
                    max_sim = sim
                    max_doc = doc_id

            return float(max_sim), max_doc

        except Exception as e:
            logger.warning(f"Semantic dedup failed: {e}")
            return 0.0, None

    def get_stats(self) -> Dict:
        """Get deduplication statistics."""
        return {
            "total_documents": len(self._seen_hashes),
            "semantic_dedup_enabled": self.enable_semantic_dedup,
            "embeddings_indexed": len(self._embeddings),
        }

    def clear(self) -> None:
        """Clear the deduplication index."""
        self._seen_hashes.clear()
        self._hash_to_doc.clear()
        self._embeddings.clear()
