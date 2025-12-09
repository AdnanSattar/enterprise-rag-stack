"""
Batch reranking utilities.

For high-throughput scenarios, batch reranking across
concurrent requests improves GPU utilization.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def deduplicate_chunks(
    results: List,
    similarity_threshold: float = 0.85,
) -> List:
    """
    Deduplicate overlapping chunks before passing to LLM.

    Failure mode: Reranker might favor overlapping chunks from
    the same document, wasting context budget on redundant info.

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


@dataclass
class BatchRequest:
    """A single reranking request in a batch."""

    request_id: str
    query: str
    candidates: List
    top_k: int = 5


@dataclass
class BatchResponse:
    """Response for a single request in batch."""

    request_id: str
    results: List
    error: Optional[str] = None


class BatchReranker:
    """
    Batch reranking across concurrent requests.

    Improves GPU utilization by batching queries across
    multiple concurrent requests.

    Usage:
        batch_reranker = BatchReranker(max_batch_size=32)
        responses = batch_reranker.rerank_batch(requests)
    """

    def __init__(
        self,
        model_name: str = None,
        max_batch_size: int = 32,
        max_workers: int = 4,
    ):
        """
        Args:
            model_name: Cross-encoder model name
            max_batch_size: Maximum candidates to process in one batch
            max_workers: Thread pool size for parallel processing
        """
        self.model_name = model_name or "cross-encoder/ms-marco-MiniLM-L-6-v2"
        self.max_batch_size = max_batch_size
        self.max_workers = max_workers
        self._model = None
        self._tokenizer = None

    def _load_model(self):
        """Load model and tokenizer."""
        if self._model is None:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            logger.info(f"Loading batch reranker model: {self.model_name}")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name
            )
            self._model.eval()

    def rerank_batch(
        self,
        requests: List[BatchRequest],
    ) -> List[BatchResponse]:
        """
        Process multiple reranking requests in batch.

        Args:
            requests: List of BatchRequest objects

        Returns:
            List of BatchResponse objects
        """
        import numpy as np
        import torch

        self._load_model()

        # Collect all query-candidate pairs
        all_pairs = []
        pair_mapping = []  # (request_idx, candidate_idx)

        for req_idx, req in enumerate(requests):
            for cand_idx, candidate in enumerate(req.candidates):
                all_pairs.append(f"{req.query} [SEP] {candidate.text}")
                pair_mapping.append((req_idx, cand_idx))

        if not all_pairs:
            return [
                BatchResponse(request_id=req.request_id, results=[]) for req in requests
            ]

        # Process in batches
        all_scores = []
        for i in range(0, len(all_pairs), self.max_batch_size):
            batch = all_pairs[i : i + self.max_batch_size]

            inputs = self._tokenizer(
                batch,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors="pt",
            )

            with torch.no_grad():
                outputs = self._model(**inputs)
                scores = outputs.logits.squeeze(-1).cpu().numpy()

            if scores.ndim == 0:
                scores = np.array([float(scores)])

            all_scores.extend(scores.tolist())

        # Group scores by request
        request_scores: Dict[int, List[tuple]] = {i: [] for i in range(len(requests))}

        for (req_idx, cand_idx), score in zip(pair_mapping, all_scores):
            request_scores[req_idx].append((cand_idx, score))

        # Build responses
        from .cross_encoder_reranker import RerankResult

        responses = []
        for req_idx, req in enumerate(requests):
            try:
                scores = request_scores[req_idx]

                if not scores:
                    responses.append(
                        BatchResponse(
                            request_id=req.request_id,
                            results=[],
                        )
                    )
                    continue

                # Normalize scores for this request
                raw_scores = [s for _, s in scores]
                min_s, max_s = min(raw_scores), max(raw_scores)

                results = []
                for cand_idx, score in scores:
                    candidate = req.candidates[cand_idx]

                    # Normalize to [0, 1]
                    if max_s - min_s > 0:
                        norm_score = (score - min_s) / (max_s - min_s)
                    else:
                        norm_score = 1.0

                    original_score = getattr(candidate, "combined_score", 0.5)
                    final_score = 0.3 * original_score + 0.7 * norm_score

                    results.append(
                        RerankResult(
                            id=candidate.id,
                            doc_id=getattr(
                                candidate, "doc_id", candidate.id.split("#")[0]
                            ),
                            text=candidate.text,
                            metadata=getattr(candidate, "metadata", {}),
                            original_score=original_score,
                            rerank_score=norm_score,
                            final_score=final_score,
                        )
                    )

                # Sort and limit
                results.sort(key=lambda x: x.final_score, reverse=True)
                results = results[: req.top_k]

                responses.append(
                    BatchResponse(
                        request_id=req.request_id,
                        results=results,
                    )
                )

            except Exception as e:
                logger.error(f"Error processing request {req.request_id}: {e}")
                responses.append(
                    BatchResponse(
                        request_id=req.request_id,
                        results=[],
                        error=str(e),
                    )
                )

        return responses
