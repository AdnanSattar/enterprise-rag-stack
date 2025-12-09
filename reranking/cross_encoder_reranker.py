"""
Cross-encoder reranking for precision boost.

Reranking is cost-effective accuracy:
- Cross-encoders are slower but more accurate than bi-encoders
- Apply only to top candidates (20-50)
- Log original scores alongside reranker scores for analysis
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

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
    
    Cross-encoders jointly encode query + document pairs,
    capturing rich interactions missed by bi-encoders.
    
    WARNING: Expensive per query - only run on 20-50 candidates!
    
    Usage:
        reranker = CrossEncoderReranker()
        results = reranker.rerank(query, candidates, top_k=5)
    """
    
    def __init__(self, model_name: str = None):
        """
        Args:
            model_name: HuggingFace model name for cross-encoder
        """
        self.model_name = model_name or "cross-encoder/ms-marco-MiniLM-L-6-v2"
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
        candidates: List,
        top_k: int = 5,
        original_weight: float = 0.3,
        rerank_weight: float = 0.7,
    ) -> List[RerankResult]:
        """
        Rerank candidates using cross-encoder.
        
        Args:
            query: Query string
            candidates: List of retrieval results (need id, doc_id, text, metadata, combined_score)
            top_k: Number of results to return
            original_weight: Weight for original score
            rerank_weight: Weight for reranker score
            
        Returns:
            Reranked results
        """
        import torch
        
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
            original_score = getattr(candidate, 'combined_score', 0.5)
            final_score = original_weight * original_score + rerank_weight * float(rerank_score)
            
            results.append(RerankResult(
                id=candidate.id,
                doc_id=getattr(candidate, 'doc_id', candidate.id.split("#")[0]),
                text=candidate.text,
                metadata=getattr(candidate, 'metadata', {}),
                original_score=original_score,
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
    Faster but less accurate than cross-encoder.
    """
    
    def __init__(self):
        self._embedding_service = None
    
    @property
    def embedding_service(self):
        """Lazy load embedding service."""
        if self._embedding_service is None:
            from embeddings import get_embedding_service
            self._embedding_service = get_embedding_service()
        return self._embedding_service
    
    def rerank(
        self,
        query: str,
        candidates: List,
        top_k: int = 5,
        original_weight: float = 0.4,
        rerank_weight: float = 0.6,
    ) -> List[RerankResult]:
        """
        Rerank using embedding cosine similarity.
        
        Args:
            query: Query string
            candidates: Retrieval results
            top_k: Number of results
            original_weight: Weight for original score
            rerank_weight: Weight for similarity score
            
        Returns:
            Reranked results
        """
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
            original_score = getattr(candidate, 'combined_score', 0.5)
            final_score = original_weight * original_score + rerank_weight * float(sim)
            
            results.append(RerankResult(
                id=candidate.id,
                doc_id=getattr(candidate, 'doc_id', candidate.id.split("#")[0]),
                text=candidate.text,
                metadata=getattr(candidate, 'metadata', {}),
                original_score=original_score,
                rerank_score=float(sim),
                final_score=final_score
            ))
        
        results.sort(key=lambda x: x.final_score, reverse=True)
        return results[:top_k]


# Global reranker instance
_reranker: Optional[CrossEncoderReranker] = None


def get_reranker(use_lightweight: bool = False):
    """
    Get reranker instance.
    
    Args:
        use_lightweight: Use fast embedding-based reranker
        
    Returns:
        Reranker instance
    """
    global _reranker
    if use_lightweight:
        return LightweightReranker()
    if _reranker is None:
        _reranker = CrossEncoderReranker()
    return _reranker


def rerank_results(
    query: str,
    candidates: List,
    top_k: int = 5,
    deduplicate: bool = True,
    use_lightweight: bool = False,
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
    from .batch_reranking import deduplicate_chunks
    
    reranker = get_reranker(use_lightweight)
    results = reranker.rerank(query, candidates, top_k)
    
    if deduplicate:
        results = deduplicate_chunks(results)
    
    return results

