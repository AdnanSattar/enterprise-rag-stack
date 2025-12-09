"""
Embedding module with versioning and drift detection.
CRITICAL: Never mix vectors from different models in the same index.
"""

import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

from .config import EmbeddingConfig, settings

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingResult:
    """Embedding result with metadata for tracking."""
    vectors: np.ndarray
    model_name: str
    model_version: str
    embedding_timestamp: str
    preprocessing_hash: str


class EmbeddingService:
    """
    Production embedding service with versioning.
    
    Key practices:
    - One embedding model version per index namespace
    - Store embedding_model and embedding_version in metadata
    - Normalize vectors to unit length for cosine similarity
    - Make embedding generation deterministic
    """
    
    def __init__(self, config: Optional[EmbeddingConfig] = None):
        self.config = config or settings.embedding
        self._model: Optional[SentenceTransformer] = None
        self._preprocessing_hash = self._compute_preprocessing_hash()
        
    def _compute_preprocessing_hash(self) -> str:
        """Hash preprocessing config for drift detection."""
        config_str = f"{self.config.model_name}:{self.config.normalize}:{self.config.version}"
        return hashlib.md5(config_str.encode()).hexdigest()[:8]
    
    @property
    def model(self) -> SentenceTransformer:
        """Lazy load the model."""
        if self._model is None:
            logger.info(f"Loading embedding model: {self.config.model_name}")
            self._model = SentenceTransformer(self.config.model_name)
        return self._model
    
    def preprocess_text(self, text: str) -> str:
        """
        Deterministic text preprocessing.
        IMPORTANT: Keep this consistent across all embeddings.
        """
        # Normalize whitespace
        text = " ".join(text.split())
        # Truncate extremely long texts
        max_chars = 8192
        if len(text) > max_chars:
            text = text[:max_chars]
        return text
    
    def embed_batch(
        self, 
        texts: List[str], 
        show_progress: bool = False
    ) -> EmbeddingResult:
        """
        Embed a batch of texts with full metadata.
        
        Args:
            texts: List of texts to embed
            show_progress: Show progress bar
            
        Returns:
            EmbeddingResult with vectors and metadata
        """
        # Preprocess all texts deterministically
        processed = [self.preprocess_text(t) for t in texts]
        
        # Generate embeddings
        vectors = self.model.encode(
            processed, 
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        
        # Normalize to unit length for cosine similarity
        if self.config.normalize:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            vectors = vectors / (norms + 1e-10)  # Avoid division by zero
        
        return EmbeddingResult(
            vectors=vectors,
            model_name=self.config.model_name,
            model_version=self.config.version,
            embedding_timestamp=datetime.utcnow().isoformat() + "Z",
            preprocessing_hash=self._preprocessing_hash
        )
    
    def embed_single(self, text: str) -> Tuple[np.ndarray, dict]:
        """Embed a single text and return vector with metadata."""
        result = self.embed_batch([text])
        metadata = {
            "embedding_model": result.model_name,
            "embedding_version": result.model_version,
            "embedding_timestamp": result.embedding_timestamp,
            "preprocessing_hash": result.preprocessing_hash,
        }
        return result.vectors[0], metadata
    
    def compute_drift_score(
        self, 
        new_vectors: np.ndarray, 
        reference_centroid: np.ndarray
    ) -> float:
        """
        Compute embedding drift score.
        
        Track average cosine similarity between new embeddings and 
        historical centroids. Alert if drift exceeds threshold.
        
        Args:
            new_vectors: New embedding vectors
            reference_centroid: Historical centroid vector
            
        Returns:
            Average cosine similarity (lower = more drift)
        """
        if len(new_vectors) == 0:
            return 1.0
            
        # Normalize vectors and centroid
        new_norms = np.linalg.norm(new_vectors, axis=1, keepdims=True)
        new_normalized = new_vectors / (new_norms + 1e-10)
        
        ref_norm = np.linalg.norm(reference_centroid)
        ref_normalized = reference_centroid / (ref_norm + 1e-10)
        
        # Compute cosine similarities
        similarities = np.dot(new_normalized, ref_normalized)
        return float(np.mean(similarities))


# Global embedding service instance
_embedding_service: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    """Get or create the global embedding service."""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service


def embed_batch(texts: List[str]) -> Tuple[np.ndarray, EmbeddingConfig]:
    """
    Convenience function for batch embedding.
    Returns vectors and config for metadata storage.
    """
    service = get_embedding_service()
    result = service.embed_batch(texts)
    return result.vectors, service.config

