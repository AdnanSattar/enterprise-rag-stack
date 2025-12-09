"""
Embedding drift detection and monitoring.

Track average cosine similarity between new embeddings and
historical centroids. Alert if drift exceeds threshold.

Drift can occur when:
- Input data distribution changes
- Preprocessing code changes unintentionally
- Model weights change (fine-tuning, updates)
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DriftAlert:
    """Alert generated when drift is detected."""

    timestamp: str
    drift_score: float
    threshold: float
    severity: str  # low, medium, high
    affected_batches: int
    message: str


@dataclass
class DriftStats:
    """Statistics for drift monitoring."""

    total_batches: int = 0
    total_vectors: int = 0
    current_drift: float = 0.0
    max_drift_observed: float = 0.0
    alerts_triggered: int = 0
    last_check: Optional[str] = None


def compute_centroid(vectors: np.ndarray) -> np.ndarray:
    """
    Compute centroid of a set of vectors.

    Args:
        vectors: Array of shape (n, dim)

    Returns:
        Centroid vector of shape (dim,)
    """
    if len(vectors) == 0:
        raise ValueError("Cannot compute centroid of empty array")

    centroid = np.mean(vectors, axis=0)

    # Normalize centroid
    norm = np.linalg.norm(centroid)
    if norm > 0:
        centroid = centroid / norm

    return centroid


def compute_drift_score(
    new_vectors: np.ndarray,
    reference_centroid: np.ndarray,
) -> float:
    """
    Compute embedding drift score.

    Lower score = more drift from reference.
    Score of 1.0 = no drift (perfect alignment).

    Args:
        new_vectors: New embedding vectors (n, dim)
        reference_centroid: Historical centroid vector (dim,)

    Returns:
        Average cosine similarity (lower = more drift)
    """
    if len(new_vectors) == 0:
        return 1.0

    # Normalize vectors
    new_norms = np.linalg.norm(new_vectors, axis=1, keepdims=True)
    new_normalized = new_vectors / (new_norms + 1e-10)

    # Normalize reference
    ref_norm = np.linalg.norm(reference_centroid)
    ref_normalized = reference_centroid / (ref_norm + 1e-10)

    # Compute cosine similarities
    similarities = np.dot(new_normalized, ref_normalized)

    return float(np.mean(similarities))


class EmbeddingDriftMonitor:
    """
    Monitor embedding drift over time.

    Usage:
        monitor = EmbeddingDriftMonitor()

        # Initialize with reference embeddings
        monitor.set_reference(initial_embeddings)

        # Check drift on new batches
        drift = monitor.check_drift(new_embeddings)
        if drift < 0.9:
            logger.warning("Significant drift detected!")

        # Get alerts
        alerts = monitor.get_alerts()
    """

    def __init__(
        self,
        drift_threshold: float = 0.90,
        alert_on_drift: bool = True,
        window_size: int = 100,
    ):
        """
        Args:
            drift_threshold: Alert when drift below this (0-1)
            alert_on_drift: Whether to generate alerts
            window_size: Number of batches for rolling stats
        """
        self.drift_threshold = drift_threshold
        self.alert_on_drift = alert_on_drift
        self.window_size = window_size

        self._reference_centroid: Optional[np.ndarray] = None
        self._drift_history: List[float] = []
        self._alerts: List[DriftAlert] = []
        self._stats = DriftStats()

    def set_reference(
        self,
        embeddings: np.ndarray,
        centroid: np.ndarray = None,
    ) -> None:
        """
        Set reference centroid for drift comparison.

        Args:
            embeddings: Reference embeddings to compute centroid from
            centroid: Pre-computed centroid (optional)
        """
        if centroid is not None:
            self._reference_centroid = centroid
        else:
            self._reference_centroid = compute_centroid(embeddings)

        logger.info(
            f"Reference centroid set from {len(embeddings)} embeddings. "
            f"Drift threshold: {self.drift_threshold}"
        )

    def check_drift(
        self,
        embeddings: np.ndarray,
        batch_id: str = None,
    ) -> float:
        """
        Check drift for a batch of embeddings.

        Args:
            embeddings: New embeddings to check
            batch_id: Optional batch identifier

        Returns:
            Drift score (1.0 = no drift, lower = more drift)
        """
        if self._reference_centroid is None:
            logger.warning("No reference centroid set. Setting from this batch.")
            self.set_reference(embeddings)
            return 1.0

        # Compute drift score
        drift_score = compute_drift_score(embeddings, self._reference_centroid)

        # Update stats
        self._stats.total_batches += 1
        self._stats.total_vectors += len(embeddings)
        self._stats.current_drift = drift_score
        self._stats.max_drift_observed = max(
            1 - drift_score, self._stats.max_drift_observed
        )
        self._stats.last_check = datetime.utcnow().isoformat() + "Z"

        # Update history
        self._drift_history.append(drift_score)
        if len(self._drift_history) > self.window_size:
            self._drift_history = self._drift_history[-self.window_size :]

        # Check for alert
        if drift_score < self.drift_threshold and self.alert_on_drift:
            self._generate_alert(drift_score, batch_id)

        return drift_score

    def _generate_alert(self, drift_score: float, batch_id: str = None) -> None:
        """Generate drift alert."""
        # Determine severity
        if drift_score < 0.7:
            severity = "high"
        elif drift_score < 0.85:
            severity = "medium"
        else:
            severity = "low"

        alert = DriftAlert(
            timestamp=datetime.utcnow().isoformat() + "Z",
            drift_score=drift_score,
            threshold=self.drift_threshold,
            severity=severity,
            affected_batches=1,
            message=(
                f"Embedding drift detected! Score: {drift_score:.3f} "
                f"(threshold: {self.drift_threshold}). "
                f"Batch: {batch_id or 'unknown'}"
            ),
        )

        self._alerts.append(alert)
        self._stats.alerts_triggered += 1

        logger.warning(alert.message)

    def get_alerts(
        self,
        since: str = None,
        severity: str = None,
    ) -> List[DriftAlert]:
        """
        Get drift alerts.

        Args:
            since: Filter alerts after this timestamp
            severity: Filter by severity level

        Returns:
            List of DriftAlert objects
        """
        alerts = self._alerts

        if severity:
            alerts = [a for a in alerts if a.severity == severity]

        if since:
            alerts = [a for a in alerts if a.timestamp > since]

        return alerts

    def get_stats(self) -> Dict:
        """Get monitoring statistics."""
        return {
            "total_batches": self._stats.total_batches,
            "total_vectors": self._stats.total_vectors,
            "current_drift": self._stats.current_drift,
            "max_drift_observed": self._stats.max_drift_observed,
            "alerts_triggered": self._stats.alerts_triggered,
            "last_check": self._stats.last_check,
            "rolling_mean_drift": (
                float(np.mean(self._drift_history)) if self._drift_history else None
            ),
            "rolling_min_drift": (
                float(np.min(self._drift_history)) if self._drift_history else None
            ),
        }

    def update_reference(
        self,
        embeddings: np.ndarray,
        alpha: float = 0.1,
    ) -> None:
        """
        Update reference centroid with exponential moving average.

        Use this to gradually adapt to legitimate distribution shifts.

        Args:
            embeddings: New embeddings
            alpha: EMA weight for new centroid (0-1)
        """
        if self._reference_centroid is None:
            self.set_reference(embeddings)
            return

        new_centroid = compute_centroid(embeddings)
        self._reference_centroid = (
            1 - alpha
        ) * self._reference_centroid + alpha * new_centroid

        # Re-normalize
        norm = np.linalg.norm(self._reference_centroid)
        if norm > 0:
            self._reference_centroid = self._reference_centroid / norm

        logger.info(f"Reference centroid updated with alpha={alpha}")

    def reset(self) -> None:
        """Reset monitor state."""
        self._reference_centroid = None
        self._drift_history = []
        self._alerts = []
        self._stats = DriftStats()
