"""
Latency metrics collection and analysis.

Tracks:
- P50, P95, P99 latencies
- Per-component latencies (retrieval, reranking, LLM)
- Request tracing
"""

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class LatencyMetrics:
    """Latency metrics for a single request."""
    
    total_ms: float
    retrieval_ms: Optional[float] = None
    reranking_ms: Optional[float] = None
    llm_ms: Optional[float] = None
    verification_ms: Optional[float] = None
    request_id: Optional[str] = None


class LatencyCollector:
    """
    Collect and analyze latency metrics.
    
    Maintains rolling windows for percentile calculations.
    """
    
    def __init__(self, window_size: int = 1000):
        """
        Args:
            window_size: Number of recent requests to keep for percentiles
        """
        self.window_size = window_size
        self.metrics: deque = deque(maxlen=window_size)
        self._component_metrics: Dict[str, deque] = {
            "retrieval": deque(maxlen=window_size),
            "reranking": deque(maxlen=window_size),
            "llm": deque(maxlen=window_size),
            "verification": deque(maxlen=window_size),
        }
    
    def record(self, metrics: LatencyMetrics):
        """Record a latency measurement."""
        self.metrics.append(metrics.total_ms)
        
        if metrics.retrieval_ms is not None:
            self._component_metrics["retrieval"].append(metrics.retrieval_ms)
        if metrics.reranking_ms is not None:
            self._component_metrics["reranking"].append(metrics.reranking_ms)
        if metrics.llm_ms is not None:
            self._component_metrics["llm"].append(metrics.llm_ms)
        if metrics.verification_ms is not None:
            self._component_metrics["verification"].append(metrics.verification_ms)
    
    def get_percentiles(self, component: Optional[str] = None) -> Dict[str, float]:
        """
        Get latency percentiles.
        
        Args:
            component: Component name (retrieval, reranking, llm, verification)
                      or None for total latency
        
        Returns:
            Dict with p50, p95, p99 values
        """
        if component:
            values = list(self._component_metrics.get(component, []))
        else:
            values = list(self.metrics)
        
        if not values:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0}
        
        sorted_values = sorted(values)
        n = len(sorted_values)
        
        return {
            "p50": sorted_values[int(n * 0.50)],
            "p95": sorted_values[int(n * 0.95)],
            "p99": sorted_values[int(n * 0.99)],
            "mean": sum(sorted_values) / n,
            "min": sorted_values[0],
            "max": sorted_values[-1],
        }
    
    def get_summary(self) -> Dict:
        """Get comprehensive latency summary."""
        summary = {
            "total": self.get_percentiles(),
            "components": {}
        }
        
        for component in self._component_metrics:
            if self._component_metrics[component]:
                summary["components"][component] = self.get_percentiles(component)
        
        return summary
    
    def reset(self):
        """Reset all metrics."""
        self.metrics.clear()
        for component in self._component_metrics:
            self._component_metrics[component].clear()


# Global latency collector
_latency_collector: Optional[LatencyCollector] = None


def get_latency_collector() -> LatencyCollector:
    """Get global latency collector."""
    global _latency_collector
    if _latency_collector is None:
        _latency_collector = LatencyCollector()
    return _latency_collector

