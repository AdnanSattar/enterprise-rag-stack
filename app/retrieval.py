"""
Hybrid retrieval module.
Retrieval is the optical lens for your LLM.

Best practices:
- Hybrid retrieval: vector + BM25 + metadata filters
- Multivector retrieval for different aspects
- Query rewrite for vague requests
- Dynamic TopK based on query complexity
"""

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import chromadb
import numpy as np
from chromadb.config import Settings as ChromaSettings

from .config import settings
from .embeddings import embed_batch, get_embedding_service

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


class VectorStore:
    """
    Vector store wrapper with Chroma backend.

    Supports:
    - HNSW index (high recall, update-friendly)
    - Metadata filtering at query time
    - Embedding versioning
    """

    def __init__(
        self,
        path: str = None,
        collection_name: str = None,
        host: str = None,
        port: int = None,
    ):
        self.path = path or settings.CHROMA_PATH
        self.collection_name = collection_name or settings.COLLECTION_NAME
        self.host = host or settings.CHROMA_HOST
        self.port = port or settings.CHROMA_PORT

        self._client: Optional[chromadb.ClientAPI] = None
        self._collection = None

    @property
    def client(self) -> chromadb.ClientAPI:
        """Get or create Chroma client."""
        if self._client is None:
            if self.host:
                # Remote client
                self._client = chromadb.HttpClient(host=self.host, port=self.port)
            else:
                # Local persistent client
                self._client = chromadb.PersistentClient(
                    path=self.path, settings=ChromaSettings(anonymized_telemetry=False)
                )
        return self._client

    @property
    def collection(self):
        """Get or create collection."""
        if self._collection is None:
            embedding_service = get_embedding_service()
            self._collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={
                    "embedding_model": embedding_service.config.model_name,
                    "embedding_version": embedding_service.config.version,
                    "hnsw:space": "cosine",
                },
            )
        return self._collection

    def add_documents(
        self,
        ids: List[str],
        texts: List[str],
        metadatas: List[Dict],
        embeddings: Optional[np.ndarray] = None,
    ) -> None:
        """
        Add documents to the vector store.

        Args:
            ids: Unique identifiers
            texts: Document texts
            metadatas: Metadata for each document
            embeddings: Pre-computed embeddings (computed if None)
        """
        if embeddings is None:
            embeddings, config = embed_batch(texts)
            # Add embedding metadata
            for meta in metadatas:
                meta["embedding_model"] = config.model_name
                meta["embedding_version"] = config.version

        self.collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings.tolist(),
        )

        logger.info(f"Added {len(ids)} documents to collection {self.collection_name}")

    def query(
        self,
        query_text: str,
        n_results: int = 10,
        where: Optional[Dict] = None,
        where_document: Optional[Dict] = None,
    ) -> Dict:
        """
        Query the vector store.

        Args:
            query_text: Query string
            n_results: Number of results to return
            where: Metadata filter
            where_document: Document content filter

        Returns:
            Chroma query results
        """
        query_embedding, _ = embed_batch([query_text])

        return self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=n_results,
            where=where,
            where_document=where_document,
            include=["documents", "metadatas", "distances"],
        )

    def delete(self, ids: List[str]) -> None:
        """Delete documents by ID."""
        self.collection.delete(ids=ids)
        logger.info(f"Deleted {len(ids)} documents")

    def count(self) -> int:
        """Get document count."""
        return self.collection.count()


def lexical_score(query: str, text: str) -> float:
    """
    Compute simple lexical overlap score (BM25-inspired).

    Pure vector retrieval fails for keyword-heavy queries
    (e.g., SKU numbers, technical codes).
    """
    q_terms = set(query.lower().split())
    t_terms = set(text.lower().split())

    if not q_terms:
        return 0.0

    overlap = len(q_terms & t_terms)

    # Simple TF-IDF inspired scoring
    return overlap / (len(q_terms) + 1e-6)


def compute_bm25_score(
    query: str, text: str, k1: float = 1.5, b: float = 0.75, avg_doc_len: float = 500
) -> float:
    """
    BM25 scoring for lexical retrieval.

    Args:
        query: Query string
        text: Document text
        k1: Term frequency saturation parameter
        b: Length normalization parameter
        avg_doc_len: Average document length in corpus
    """
    q_terms = query.lower().split()
    doc_terms = text.lower().split()
    doc_len = len(doc_terms)

    if doc_len == 0:
        return 0.0

    # Term frequency in document
    tf = {}
    for term in doc_terms:
        tf[term] = tf.get(term, 0) + 1

    score = 0.0
    for term in q_terms:
        if term in tf:
            freq = tf[term]
            # BM25 formula
            numerator = freq * (k1 + 1)
            denominator = freq + k1 * (1 - b + b * (doc_len / avg_doc_len))
            score += numerator / denominator

    return score


class HybridRetriever:
    """
    Hybrid retrieval combining vector + lexical + metadata.

    Score fusion: score = w_vec * s_vec + w_bm25 * s_bm25
    """

    def __init__(
        self,
        vector_store: VectorStore = None,
        vector_weight: float = None,
        lexical_weight: float = None,
    ):
        self.vector_store = vector_store or VectorStore()
        self.vector_weight = vector_weight or settings.retrieval.vector_weight
        self.lexical_weight = lexical_weight or settings.retrieval.lexical_weight

    def retrieve(
        self,
        query: str,
        top_k: int = None,
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
            Ranked list of retrieval results
        """
        top_k = top_k or settings.retrieval.default_top_k

        # 1. Vector search for semantic candidates
        # Get more candidates than needed for fusion
        vector_results = self.vector_store.query(
            query_text=query, n_results=top_k * 2, where=filters
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
            # Chroma returns L2 distance for cosine space, convert to similarity
            vector_score = 1 - (dist / 2)  # Approximate conversion

            # Lexical score
            if use_bm25:
                lex_score = compute_bm25_score(query, text)
            else:
                lex_score = lexical_score(query, text)

            # Normalize lexical score to [0, 1] range
            lex_score = min(lex_score / 5.0, 1.0)

            # Weighted fusion
            combined = (
                self.vector_weight * vector_score + self.lexical_weight * lex_score
            )

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

    def retrieve_with_query_expansion(
        self, query: str, top_k: int = None, filters: Optional[Dict] = None
    ) -> List[RetrievalResult]:
        """
        Retrieval with simple query expansion.

        Canonicalize user query before embedding to improve recall.
        """
        # Simple query canonicalization
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
_vector_store: Optional[VectorStore] = None
_retriever: Optional[HybridRetriever] = None


def get_vector_store() -> VectorStore:
    """Get global vector store instance."""
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store


def get_retriever() -> HybridRetriever:
    """Get global retriever instance."""
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever


def add_documents(ids: List[str], texts: List[str], metadatas: List[Dict]) -> None:
    """Add documents to vector store."""
    get_vector_store().add_documents(ids, texts, metadatas)


def query_collection(
    query: str, n_results: int = 10, where: Optional[Dict] = None
) -> Dict:
    """Query the collection directly."""
    return get_vector_store().query(query, n_results, where)
