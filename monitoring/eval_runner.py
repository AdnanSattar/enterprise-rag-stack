"""
Monitoring and evaluation module.
What you cannot measure you cannot improve.

Metrics to collect:
- Recall@k and reranker precision
- Hallucination rate
- Latency p50, p95, p99
- Cost per request and token usage
- Embedding drift statistics
"""

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EvalItem:
    """A single evaluation item."""

    query: str
    expected_doc_ids: List[str]
    expected_answer: Optional[str] = None
    scorer: Optional[Callable] = None
    tags: List[str] = field(default_factory=list)


@dataclass
class EvalResult:
    """Result from a single evaluation."""

    query: str
    score: float
    latency_ms: float
    retrieved_ids: List[str]
    recall: float
    precision: float
    details: Dict = field(default_factory=dict)


@dataclass
class EvalRunSummary:
    """Summary of an evaluation run."""

    run_id: str
    timestamp: str
    total_queries: int
    mean_score: float
    mean_recall: float
    mean_precision: float
    p25_score: float
    p50_score: float
    p75_score: float
    p95_latency_ms: float
    p99_latency_ms: float
    results: List[EvalResult]


def compute_recall_at_k(
    retrieved_ids: List[str], expected_ids: List[str], k: int = None
) -> float:
    """
    Compute Recall@K.

    What fraction of expected documents were retrieved?
    """
    if not expected_ids:
        return 1.0

    if k:
        retrieved_ids = retrieved_ids[:k]

    retrieved_set = set(retrieved_ids)
    expected_set = set(expected_ids)

    found = len(retrieved_set & expected_set)
    return found / len(expected_set)


def compute_precision_at_k(
    retrieved_ids: List[str], expected_ids: List[str], k: int = None
) -> float:
    """
    Compute Precision@K.

    What fraction of retrieved documents were relevant?
    """
    if k:
        retrieved_ids = retrieved_ids[:k]

    if not retrieved_ids:
        return 0.0

    retrieved_set = set(retrieved_ids)
    expected_set = set(expected_ids)

    relevant = len(retrieved_set & expected_set)
    return relevant / len(retrieved_set)


def compute_mrr(retrieved_ids: List[str], expected_ids: List[str]) -> float:
    """
    Compute Mean Reciprocal Rank.

    Where does the first relevant document appear?
    """
    expected_set = set(expected_ids)

    for i, doc_id in enumerate(retrieved_ids):
        if doc_id in expected_set:
            return 1.0 / (i + 1)

    return 0.0


def compute_ndcg(
    retrieved_ids: List[str], expected_ids: List[str], k: int = None
) -> float:
    """
    Compute Normalized Discounted Cumulative Gain.
    """
    if k:
        retrieved_ids = retrieved_ids[:k]

    expected_set = set(expected_ids)

    # Compute DCG
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_ids):
        if doc_id in expected_set:
            dcg += 1.0 / np.log2(i + 2)  # +2 because log2(1) = 0

    # Compute ideal DCG
    ideal_dcg = sum(
        1.0 / np.log2(i + 2) for i in range(min(len(expected_ids), len(retrieved_ids)))
    )

    if ideal_dcg == 0:
        return 0.0

    return dcg / ideal_dcg


class RAGEvaluator:
    """
    Evaluation framework for RAG pipelines.

    Maintains a golden evaluation set and tracks trends.
    """

    def __init__(self, pipeline: Callable):
        """
        Args:
            pipeline: Function that takes a query and returns
                      {"retrieved_ids": [...], "answer": "..."}
        """
        self.pipeline = pipeline
        self.eval_history: List[EvalRunSummary] = []

    def evaluate_single(self, item: EvalItem, k: int = 10) -> EvalResult:
        """Evaluate a single query."""
        start_time = time.time()

        # Run pipeline
        result = self.pipeline(item.query)

        latency_ms = (time.time() - start_time) * 1000
        retrieved_ids = result.get("retrieved_ids", [])

        # Compute metrics
        recall = compute_recall_at_k(retrieved_ids, item.expected_doc_ids, k)
        precision = compute_precision_at_k(retrieved_ids, item.expected_doc_ids, k)
        mrr = compute_mrr(retrieved_ids, item.expected_doc_ids)

        # Compute overall score
        if item.scorer:
            score = item.scorer(result, item.expected_answer)
        else:
            # Default: weighted average of recall and precision
            score = 0.6 * recall + 0.4 * precision

        return EvalResult(
            query=item.query,
            score=score,
            latency_ms=latency_ms,
            retrieved_ids=retrieved_ids[:k],
            recall=recall,
            precision=precision,
            details={
                "mrr": mrr,
                "expected_ids": item.expected_doc_ids,
                "tags": item.tags,
            },
        )

    def run_evaluation(
        self, eval_set: List[EvalItem], run_id: str = None, k: int = 10
    ) -> EvalRunSummary:
        """
        Run full evaluation on an evaluation set.

        Args:
            eval_set: List of evaluation items
            run_id: Identifier for this run
            k: Cutoff for recall/precision

        Returns:
            EvalRunSummary with all metrics
        """
        run_id = run_id or datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        results = []

        logger.info(f"Starting evaluation run {run_id} with {len(eval_set)} queries")

        for item in eval_set:
            result = self.evaluate_single(item, k)
            results.append(result)

        # Compute summary statistics
        scores = [r.score for r in results]
        recalls = [r.recall for r in results]
        precisions = [r.precision for r in results]
        latencies = [r.latency_ms for r in results]

        summary = EvalRunSummary(
            run_id=run_id,
            timestamp=datetime.utcnow().isoformat() + "Z",
            total_queries=len(eval_set),
            mean_score=float(np.mean(scores)),
            mean_recall=float(np.mean(recalls)),
            mean_precision=float(np.mean(precisions)),
            p25_score=float(np.percentile(scores, 25)),
            p50_score=float(np.percentile(scores, 50)),
            p75_score=float(np.percentile(scores, 75)),
            p95_latency_ms=float(np.percentile(latencies, 95)),
            p99_latency_ms=float(np.percentile(latencies, 99)),
            results=results,
        )

        self.eval_history.append(summary)

        logger.info(
            f"Evaluation complete: mean_recall={summary.mean_recall:.3f}, "
            f"mean_precision={summary.mean_precision:.3f}, "
            f"p95_latency={summary.p95_latency_ms:.1f}ms"
        )

        return summary

    def compare_runs(self, run_id_a: str, run_id_b: str) -> Dict:
        """Compare two evaluation runs."""
        run_a = next((r for r in self.eval_history if r.run_id == run_id_a), None)
        run_b = next((r for r in self.eval_history if r.run_id == run_id_b), None)

        if not run_a or not run_b:
            return {"error": "Run not found"}

        return {
            "run_a": run_id_a,
            "run_b": run_id_b,
            "recall_delta": run_b.mean_recall - run_a.mean_recall,
            "precision_delta": run_b.mean_precision - run_a.mean_precision,
            "score_delta": run_b.mean_score - run_a.mean_score,
            "latency_delta_p95": run_b.p95_latency_ms - run_a.p95_latency_ms,
            "improved": run_b.mean_score > run_a.mean_score,
        }


def run_eval(eval_set: List[Dict], pipeline: Callable) -> Dict:
    """
    Convenience function for quick evaluation.

    Args:
        eval_set: List of {"query": str, "expected_ids": [...], "scorer": fn}
        pipeline: RAG pipeline function

    Returns:
        Summary statistics
    """
    results = []

    for item in eval_set:
        start = time.time()
        resp = pipeline(item["query"])
        latency = (time.time() - start) * 1000

        if "scorer" in item:
            score = item["scorer"](resp, item.get("expected"))
        else:
            # Default scoring based on retrieval
            retrieved = resp.get("retrieved_ids", [])
            expected = item.get("expected_ids", [])
            score = compute_recall_at_k(retrieved, expected)

        results.append({"score": score, "latency_ms": latency})

    scores = [r["score"] for r in results]
    latencies = [r["latency_ms"] for r in results]

    return {
        "mean": float(np.mean(scores)),
        "p25": float(np.percentile(scores, 25)),
        "p50": float(np.percentile(scores, 50)),
        "p75": float(np.percentile(scores, 75)),
        "p95_latency_ms": float(np.percentile(latencies, 95)),
        "total_queries": len(eval_set),
    }


class MetricsCollector:
    """
    Collect operational metrics for monitoring.

    Export to Prometheus/Grafana or log to structured logs.
    """

    def __init__(self):
        self.metrics: Dict[str, List] = {
            "query_latency_ms": [],
            "retrieval_latency_ms": [],
            "llm_latency_ms": [],
            "tokens_used": [],
            "chunks_retrieved": [],
            "cache_hits": [],
        }

    def record(self, metric_name: str, value: float):
        """Record a metric value."""
        if metric_name in self.metrics:
            self.metrics[metric_name].append(value)

    def get_summary(self) -> Dict:
        """Get summary of all metrics."""
        summary = {}
        for name, values in self.metrics.items():
            if values:
                summary[name] = {
                    "count": len(values),
                    "mean": float(np.mean(values)),
                    "p50": float(np.percentile(values, 50)),
                    "p95": float(np.percentile(values, 95)),
                    "p99": float(np.percentile(values, 99)),
                }
        return summary

    def reset(self):
        """Reset all metrics."""
        for key in self.metrics:
            self.metrics[key] = []


# Global metrics collector
_metrics = MetricsCollector()


def get_metrics() -> MetricsCollector:
    """Get global metrics collector."""
    return _metrics
