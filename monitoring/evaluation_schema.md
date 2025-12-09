# Evaluation Schema

This document describes the schema for evaluation runs and golden datasets.

## Golden Evaluation Set Format

```json
{
  "version": "1.0",
  "created_at": "2025-01-01T00:00:00Z",
  "items": [
    {
      "query": "What are the payment terms?",
      "expected_doc_ids": ["contract_001#chunk_0", "contract_001#chunk_1"],
      "expected_answer": "Payment must be made within 30 days...",
      "tags": ["legal", "contract"],
      "scorer": "exact_match"
    }
  ]
}
```

## Evaluation Run Schema

```json
{
  "run_id": "20250101_120000",
  "timestamp": "2025-01-01T12:00:00Z",
  "pipeline_version": "v1.0.0",
  "config": {
    "top_k": 10,
    "rerank": true,
    "verify": true
  },
  "results": [
    {
      "query": "What are the payment terms?",
      "retrieved_ids": ["contract_001#chunk_0", "contract_002#chunk_5"],
      "answer": "Payment must be made within 30 days...",
      "metrics": {
        "recall_at_10": 0.5,
        "precision_at_10": 0.5,
        "mrr": 1.0,
        "latency_ms": 234.5
      }
    }
  ],
  "summary": {
    "mean_recall": 0.85,
    "mean_precision": 0.70,
    "p95_latency_ms": 1234.5
  }
}
```

## Database Schema (Optional)

```sql
CREATE TABLE evaluation_runs (
    run_id VARCHAR(50) PRIMARY KEY,
    timestamp TIMESTAMP,
    pipeline_version VARCHAR(20),
    config JSONB,
    summary JSONB
);

CREATE TABLE evaluation_results (
    id SERIAL PRIMARY KEY,
    run_id VARCHAR(50) REFERENCES evaluation_runs(run_id),
    query TEXT,
    retrieved_ids TEXT[],
    answer TEXT,
    metrics JSONB
);
```
