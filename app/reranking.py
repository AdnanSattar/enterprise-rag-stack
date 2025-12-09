"""
Reranking pipeline.
Reranking is cost-effective accuracy.

Key points:
- Cross-encoder reranker for top candidates
- Merge semantic and lexical signals
- Deduplicate overlapping chunks
- Only run on 20-50 candidates (expensive per query)
"""

import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import settings
from .retrieval import RetrievalResult

logger = logging.getLogger(__name__)


@dataclass
class RerankResult:
    """Reranked result with scores."""
    id: str
    doc_id: str
    text: str
    metadata: Dict
    original_score: float
    rerank_score: float
    final_score: float


class CrossEncoderReranker:
    """
    Cross-encoder reranker using HuggingFace transformers.
    
    Cross-encoders are slower but more accurate than bi-encoders.
    Apply only to top candidates (20-50).
    """
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.retrieval.reranker_model
        self._tokenizer = None
        self._model = None
    
    @property
    def tokenizer(self):
        """Lazy load tokenizer."""
        if self._tokenizer is None:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        return self._tokenizer
    
    @property
    def model(self):
        """Lazy load model."""
        if self._model is None:
            from transformers import AutoModelForSequenceClassification
            logger.info(f"Loading reranker model: {self.model_name}")
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name
            )
            self._model.eval()
        return self._model
    
    def rerank(
        self,
        query: str,
        candidates: List[RetrievalResult],
        top_k: int = None
    ) -> List[RerankResult]:
        """
        Rerank candidates using cross-encoder.
        
        Args:
            query: Query string
            candidates: List of retrieval results
            top_k: Number of results to return
            
        Returns:
            Reranked results
        """
        import torch
        
        top_k = top_k or settings.retrieval.rerank_top_k
        
        if not candidates:
            return []
        
        # Prepare inputs: query [SEP] candidate
        texts = [f"{query} [SEP] {c.text}" for c in candidates]
        
        # Batch tokenize
        inputs = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        )
        
        # Get scores
        with torch.no_grad():
            outputs = self.model(**inputs)
            scores = outputs.logits.squeeze(-1).cpu().numpy()
        
        # Handle single result case
        if scores.ndim == 0:
            scores = np.array([float(scores)])
        
        # Normalize scores to [0, 1]
        if len(scores) > 1:
            scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-6)
        else:
            scores = np.array([1.0])
        
        # Combine with original scores
        results = []
        for candidate, rerank_score in zip(candidates, scores):
            # Weight original and rerank scores
            final_score = 0.3 * candidate.combined_score + 0.7 * float(rerank_score)
            
            results.append(RerankResult(
                id=candidate.id,
                doc_id=candidate.doc_id,
                text=candidate.text,
                metadata=candidate.metadata,
                original_score=candidate.combined_score,
                rerank_score=float(rerank_score),
                final_score=final_score
            ))
        
        # Sort by final score
        results.sort(key=lambda x: x.final_score, reverse=True)
        
        # Log score distribution for analysis
        logger.debug(
            f"Rerank scores - min: {min(r.rerank_score for r in results):.3f}, "
            f"max: {max(r.rerank_score for r in results):.3f}"
        )
        
        return results[:top_k]


class LightweightReranker:
    """
    Lightweight reranker using embedding similarity.
    Use when cross-encoder is too slow.
    """
    
    def __init__(self):
        from .embeddings import get_embedding_service
        self.embedding_service = get_embedding_service()
    
    def rerank(
        self,
        query: str,
        candidates: List[RetrievalResult],
        top_k: int = None
    ) -> List[RerankResult]:
        """Rerank using embedding cosine similarity."""
        top_k = top_k or settings.retrieval.rerank_top_k
        
        if not candidates:
            return []
        
        # Get query embedding
        query_result = self.embedding_service.embed_batch([query])
        query_vec = query_result.vectors[0]
        
        # Get candidate embeddings
        texts = [c.text for c in candidates]
        candidate_result = self.embedding_service.embed_batch(texts)
        candidate_vecs = candidate_result.vectors
        
        # Compute cosine similarities
        similarities = np.dot(candidate_vecs, query_vec)
        
        results = []
        for candidate, sim in zip(candidates, similarities):
            final_score = 0.4 * candidate.combined_score + 0.6 * float(sim)
            
            results.append(RerankResult(
                id=candidate.id,
                doc_id=candidate.doc_id,
                text=candidate.text,
                metadata=candidate.metadata,
                original_score=candidate.combined_score,
                rerank_score=float(sim),
                final_score=final_score
            ))
        
        results.sort(key=lambda x: x.final_score, reverse=True)
        return results[:top_k]


def deduplicate_chunks(
    results: List[RerankResult],
    similarity_threshold: float = 0.85
) -> List[RerankResult]:
    """
    Deduplicate overlapping chunks before passing to LLM.
    
    Args:
        results: Reranked results
        similarity_threshold: Jaccard similarity threshold for dedup
        
    Returns:
        Deduplicated results
    """
    if len(results) <= 1:
        return results
    
    def jaccard_similarity(text1: str, text2: str) -> float:
        """Compute word-level Jaccard similarity."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
            
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        return intersection / union
    
    deduplicated = [results[0]]
    
    for candidate in results[1:]:
        is_duplicate = False
        for kept in deduplicated:
            if jaccard_similarity(candidate.text, kept.text) > similarity_threshold:
                is_duplicate = True
                logger.debug(f"Removing duplicate chunk: {candidate.id}")
                break
        
        if not is_duplicate:
            deduplicated.append(candidate)
    
    return deduplicated


# Global reranker instance
_reranker: Optional[CrossEncoderReranker] = None


def get_reranker(use_lightweight: bool = False):
    """Get reranker instance."""
    global _reranker
    if use_lightweight:
        return LightweightReranker()
    if _reranker is None:
        _reranker = CrossEncoderReranker()
    return _reranker


def rerank_results(
    query: str,
    candidates: List[RetrievalResult],
    top_k: int = None,
    deduplicate: bool = True,
    use_lightweight: bool = False
) -> List[RerankResult]:
    """
    Convenience function for reranking.
    
    Args:
        query: Query string
        candidates: Retrieval results
        top_k: Number of results
        deduplicate: Remove overlapping chunks
        use_lightweight: Use fast embedding-based reranker
        
    Returns:
        Reranked and optionally deduplicated results
    """
    reranker = get_reranker(use_lightweight)
    results = reranker.rerank(query, candidates, top_k)
    
    if deduplicate:
        results = deduplicate_chunks(results)
    
    return results

