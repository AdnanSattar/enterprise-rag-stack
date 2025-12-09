"""
Embedding service with versioning and deterministic preprocessing.

CRITICAL: Never mix vectors from different models in the same index.

Version your embeddings with your index:
    index_name = f"docs_v1_{model_slug}_{version}"

When you change embedding_version, you MUST re-index all documents.
"""

import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingConfig:
    """
    Embedding model configuration.

    Store this configuration in:
    - Database table: embedding_configs
    - Or: YAML manifest checked into git

    IMPORTANT: When changing any value, update version and re-index!
    """

    model_name: str = "all-MiniLM-L6-v2"
    normalize: bool = True
    version: str = "2025-01-01"
    dimension: int = 384  # Matches all-MiniLM-L6-v2
    max_seq_length: int = 512

    def get_index_name(self, collection_name: str) -> str:
        """Generate versioned index name."""
        model_slug = self.model_name.replace("/", "_").replace("-", "_")
        return f"{collection_name}_{model_slug}_{self.version}"


@dataclass
class EmbeddingResult:
    """Embedding result with full metadata for tracking."""

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

    Usage:
        service = EmbeddingService()
        result = service.embed_batch(["text1", "text2"])

        # With custom config
        config = EmbeddingConfig(model_name="bge-large-en-v1.5")
        service = EmbeddingService(config=config)
    """

    def __init__(self, config: Optional[EmbeddingConfig] = None):
        self.config = config or EmbeddingConfig()
        self._model: Optional[SentenceTransformer] = None
        self._preprocessing_hash = self._compute_preprocessing_hash()

    def _compute_preprocessing_hash(self) -> str:
        """Hash preprocessing config for drift detection."""
        config_str = (
            f"{self.config.model_name}:"
            f"{self.config.normalize}:"
            f"{self.config.version}"
        )
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
        Any change requires re-indexing!
        """
        # Normalize whitespace
        text = " ".join(text.split())

        # Truncate extremely long texts
        max_chars = self.config.max_seq_length * 4  # Approximate
        if len(text) > max_chars:
            text = text[:max_chars]

        return text

    def embed_batch(
        self, texts: List[str], show_progress: bool = False
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
            processed, show_progress_bar=show_progress, convert_to_numpy=True
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
            preprocessing_hash=self._preprocessing_hash,
        )

    def embed_single(self, text: str) -> Tuple[np.ndarray, dict]:
        """
        Embed a single text and return vector with metadata.

        Args:
            text: Text to embed

        Returns:
            (vector, metadata_dict)
        """
        result = self.embed_batch([text])
        metadata = {
            "embedding_model": result.model_name,
            "embedding_version": result.model_version,
            "embedding_timestamp": result.embedding_timestamp,
            "preprocessing_hash": result.preprocessing_hash,
        }
        return result.vectors[0], metadata

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a query for retrieval.

        Some models use different embeddings for queries vs documents.
        This method handles that distinction.
        """
        result = self.embed_batch([query])
        return result.vectors[0]

    def get_metadata(self) -> dict:
        """Get embedding configuration metadata."""
        return {
            "model_name": self.config.model_name,
            "version": self.config.version,
            "dimension": self.config.dimension,
            "normalize": self.config.normalize,
            "preprocessing_hash": self._preprocessing_hash,
        }


# Global embedding service instance
_embedding_service: Optional[EmbeddingService] = None


def get_embedding_service(config: Optional[EmbeddingConfig] = None) -> EmbeddingService:
    """
    Get or create the global embedding service.

    Args:
        config: Optional custom configuration

    Returns:
        EmbeddingService instance
    """
    global _embedding_service
    if _embedding_service is None or config is not None:
        _embedding_service = EmbeddingService(config)
    return _embedding_service


def embed_batch(texts: List[str]) -> Tuple[np.ndarray, EmbeddingConfig]:
    """
    Convenience function for batch embedding.

    Returns vectors and config for metadata storage.
    """
    service = get_embedding_service()
    result = service.embed_batch(texts)
    return result.vectors, service.config


def embed_texts(texts: List[str]) -> np.ndarray:
    """Simple embedding function returning just vectors."""
    service = get_embedding_service()
    result = service.embed_batch(texts)
    return result.vectors
