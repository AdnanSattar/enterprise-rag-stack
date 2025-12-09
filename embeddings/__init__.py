"""
Embeddings Module.

CRITICAL: Never mix vectors from different models in the same index.

This module handles:
- Embedding generation with versioning
- Drift detection and monitoring
- Preprocessing normalization
- Compatibility projection (research)

Key practices:
- One embedding model version per index namespace
- Store embedding_model and embedding_version in metadata
- Normalize vectors to unit length for cosine similarity
- Make embedding generation deterministic

Usage:
    from embeddings import EmbeddingService, get_embedding_service

    service = get_embedding_service()
    result = service.embed_batch(["text1", "text2"])
"""

from .drift_monitor import (
    DriftAlert,
    EmbeddingDriftMonitor,
    compute_centroid,
    compute_drift_score,
)
from .embedder import (
    EmbeddingConfig,
    EmbeddingResult,
    EmbeddingService,
    embed_batch,
    get_embedding_service,
)

__all__ = [
    "EmbeddingService",
    "EmbeddingResult",
    "EmbeddingConfig",
    "get_embedding_service",
    "embed_batch",
    "EmbeddingDriftMonitor",
    "DriftAlert",
    "compute_centroid",
    "compute_drift_score",
]
