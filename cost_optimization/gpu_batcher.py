"""
GPU batching for efficient embedding and reranking.

Batches multiple requests together to maximize GPU utilization.
"""

import time
from dataclasses import dataclass
from queue import Queue
from threading import Thread
from typing import Callable, List, Optional


@dataclass
class BatchItem:
    """Item in a batch."""

    data: any
    callback: Callable
    timestamp: float


class GPUBatcher:
    """
    Batch processing for GPU operations.

    Collects requests over a time window and processes them together.
    """

    def __init__(
        self,
        batch_size: int = 32,
        max_wait_ms: float = 50.0,
        process_fn: Optional[Callable] = None,
    ):
        """
        Args:
            batch_size: Maximum items per batch
            max_wait_ms: Maximum wait time before processing batch
            process_fn: Function to process batches
        """
        self.batch_size = batch_size
        self.max_wait_ms = max_wait_ms
        self.process_fn = process_fn

        self.queue: Queue = Queue()
        self.batch: List[BatchItem] = []
        self.last_batch_time = time.time()

        self._thread: Optional[Thread] = None
        self._running = False

    def add(self, data: any, callback: Callable):
        """Add item to batch."""
        item = BatchItem(data=data, callback=callback, timestamp=time.time())
        self.queue.put(item)

    def _process_batch(self, batch: List[BatchItem]):
        """Process a batch of items."""
        if not batch:
            return

        batch_data = [item.data for item in batch]

        if self.process_fn:
            results = self.process_fn(batch_data)
        else:
            # Default: process individually
            results = [self._default_process(item.data) for item in batch]

        # Call callbacks
        for item, result in zip(batch, results):
            try:
                item.callback(result)
            except Exception as e:
                print(f"Callback error: {e}")

    def _default_process(self, data: any) -> any:
        """Default processing function."""
        return data

    def _worker(self):
        """Worker thread that processes batches."""
        while self._running:
            try:
                # Wait for item or timeout
                item = self.queue.get(timeout=self.max_wait_ms / 1000.0)
                self.batch.append(item)

                # Check if batch is full or timeout reached
                elapsed_ms = (time.time() - self.last_batch_time) * 1000
                if len(self.batch) >= self.batch_size or elapsed_ms >= self.max_wait_ms:
                    self._process_batch(self.batch)
                    self.batch = []
                    self.last_batch_time = time.time()

            except:
                # Timeout or queue empty - process current batch
                if self.batch:
                    self._process_batch(self.batch)
                    self.batch = []
                    self.last_batch_time = time.time()

    def start(self):
        """Start the batcher."""
        if self._running:
            return

        self._running = True
        self._thread = Thread(target=self._worker, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the batcher."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)

        # Process remaining batch
        if self.batch:
            self._process_batch(self.batch)
            self.batch = []


def batch_embeddings(texts: List[str], batch_size: int = 32) -> List[List[float]]:
    """
    Batch process embeddings.

    Args:
        texts: List of texts to embed
        batch_size: Batch size

    Returns:
        List of embedding vectors
    """
    # This would integrate with your embedding service
    # For now, placeholder
    return [[] for _ in texts]
