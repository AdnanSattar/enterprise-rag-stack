# Enterprise RAG Stack

<div align="center">

**A complete, production-grade reference implementation of modern Retrieval-Augmented Generation systems for 2025 and beyond.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

</div>

---

This repository accompanies the article series **"The Enterprise RAG Engineering Playbook (2025 Edition)"** and provides modular, end-to-end components for building high-accuracy, scalable, and secure RAG pipelines.

## 🎯 Use Cases

Build production-grade systems for:

- **Enterprise Search** — Find information across thousands of documents
- **Compliance Assistants** — Policy Q&A with audit trails
- **Legal Reasoning Tools** — Contract analysis with citation
- **Customer Support Copilots** — Grounded answers from knowledge bases
- **Multi-hop Knowledge Systems** — Complex queries requiring graph traversal

---

## ✨ Features

This repository includes reference implementations of **all core RAG subsystems**:

| # | Module | Description |
|---|--------|-------------|
| 1 | **Document Ingestion** | OCR cleanup, boilerplate removal, metadata extraction |
| 2 | **Semantic Chunking** | Title-aware, section-aware, token-budget segmentation |
| 3 | **Embeddings** | Versioning, drift detection, deterministic preprocessing |
| 4 | **Hybrid Retrieval** | Vector (HNSW) + BM25 + metadata filtering + score fusion |
| 5 | **Reranking** | Cross-encoder reranking, batch inference, deduplication |
| 6 | **Context Assembly** | Structured prompts, token budgeting, inline citations |
| 7 | **Verification** | Deterministic checks + LLM critic validation |
| 8 | **Graph-RAG** | Entity linking, knowledge graph traversal, multi-hop evidence |
| 9 | **Monitoring** | Recall@k, latency dashboards, drift detection |
| 10 | **Deployment** | FastAPI, circuit breakers, Docker, Kubernetes |
| 11 | **Security** | ACL filtering, PII redaction, audit logging |
| 12 | **Cost Optimization** | Model routing, caching, GPU batching |

---

## 📁 Repository Structure

```
enterprise-rag-stack/
│
├── README.md
├── LICENSE
├── requirements.txt
│
├── docs/
│   ├── architecture_overview.md
│   ├── diagrams/
│   └── article_series_index.md
│
├── config/
│   ├── embedding_config.yaml
│   ├── retrieval_config.yaml
│   ├── model_routing.yaml
│   ├── index_config.yaml
│   ├── reranker_config.yaml
│   └── logging.yaml
│
├── data/
│   ├── sample_documents/
│   ├── golden_eval_set.json
│   └── pii_redaction_rules.json
│
├── ingestion/              # Document normalization pipeline
│   ├── normalize.py
│   ├── ingest_pipeline.py
│   └── dedupe.py
│
├── chunking/               # Semantic chunking strategies
│   ├── semantic_chunker.py
│   ├── sentence_splitter.py
│   └── chunk_eval_tools.py
│
├── embeddings/             # Embedding with versioning
│   ├── embedder.py
│   └── drift_monitor.py
│
├── retrieval/              # Hybrid retrieval system
│   ├── hybrid_retriever.py
│   ├── vector_retriever.py
│   ├── lexical_retriever.py
│   └── score_fusion.py
│
├── reranking/              # Cross-encoder reranking
│   ├── cross_encoder_reranker.py
│   └── batch_reranking.py
│
├── context/                # Context assembly & prompts
│   ├── context_builder.py
│   ├── context_budgeting.py
│   └── prompt_templates/
│
├── verification/           # Answer verification
│   ├── deterministic_checks.py
│   ├── critic_llm.py
│   └── verification_pipeline.py
│
├── graph_rag/              # Knowledge graph integration
│   ├── entity_linking.py
│   ├── graph_builder.py
│   ├── graph_traversal.py
│   └── kg_summarizer.py
│
├── monitoring/             # Evaluation & metrics
│   ├── eval_runner.py
│   ├── recall_metrics.py
│   ├── latency_metrics.py
│   └── evaluation_schema.md
│
├── deployment/             # FastAPI, Docker, K8s
│   ├── fastapi_app.py
│   ├── circuit_breaker.py
│   ├── docker/
│   │   ├── Dockerfile
│   │   └── docker-compose.yaml
│   └── k8s/
│
├── security/               # ACLs, PII, audit
│   ├── pii_redaction.py
│   ├── acl_filters.py
│   └── audit_logging.py
│
└── cost_optimization/      # Model routing, caching
    ├── model_router.py
    ├── caching_layer.py
    ├── gpu_batcher.py
    └── unit_economics_calculator.py
```

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/AdnanSattar/enterprise-rag-stack.git
cd enterprise-rag-stack

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export OPENAI_API_KEY="your-api-key"
export CHROMA_PATH="./data/chroma"
```

### Run Sample Pipeline

```bash
# Ingest sample documents
python ingestion/ingest_pipeline.py --input data/sample_documents --tenant default

# Run hybrid retriever
python retrieval/hybrid_retriever.py --query "What are the payment terms?"
```

### Start the API Server

```bash
# Run FastAPI server
python deployment/fastapi_app.py

# Or with uvicorn
uvicorn deployment.fastapi_app:app --reload --port 8000
```

### Docker Deployment

```bash
# Build and start all services
docker-compose up -d

# Check logs
docker-compose logs -f rag-api
```

---

## 📖 Key Concepts

### The High-Accuracy RAG Formula

```
Accuracy = Quality(Ingestion) × Recall(Retrieval) × Precision(Reranking) × Grounding(Verification)
```

### Hybrid Retrieval Score Fusion

```python
score = vector_weight × s_vector + lexical_weight × s_bm25
```

**Why hybrid?** Pure vector fails for keyword-heavy queries (SKU numbers, codes). BM25 captures exact matches.

### Two-Level Verification

| Level | Method | What It Catches |
|-------|--------|-----------------|
| 1 | Deterministic | Numbers, dates, currencies not in context |
| 2 | LLM Critic | Unsupported claims, hallucinations |

### Embedding Versioning

```python
# CRITICAL: When model changes, re-index everything
index_name = f"docs_v1_{model_slug}_{embedding_version}"
```

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                      Enterprise RAG Stack                             │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│   ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐              │
│   │ Ingest  │──▶│  Chunk  │──▶│  Embed  │──▶│  Index  │              │
│   │ & Clean │   │Semantic │   │ Version │   │  HNSW   │              │
│   └─────────┘   └─────────┘   └─────────┘   └─────────┘              │
│                                                                       │
│   ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐              │
│   │ Query   │──▶│ Hybrid  │──▶│ Rerank  │──▶│ Context │              │
│   │Classify │   │Retrieve │   │ Top-K   │   │Assemble │              │
│   └─────────┘   └─────────┘   └─────────┘   └─────────┘              │
│                                                                       │
│   ┌─────────┐   ┌─────────┐   ┌─────────┐                            │
│   │   LLM   │──▶│ Verify  │──▶│Response │                            │
│   │Generate │   │ Answer  │   │ + Cache │                            │
│   └─────────┘   └─────────┘   └─────────┘                            │
│                                                                       │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Performance Targets

| Metric | Target | Alert Threshold |
|--------|--------|-----------------|
| Recall@5 | > 0.85 | < 0.75 |
| Precision@5 | > 0.70 | < 0.60 |
| P95 Latency | < 2000ms | > 5000ms |
| Hallucination Rate | < 5% | > 10% |
| Cache Hit Rate | > 40% | < 20% |

---

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | Required |
| `CHROMA_PATH` | ChromaDB storage path | `./data/chroma` |
| `COLLECTION_NAME` | Vector collection name | `docs_v1` |
| `REDIS_URL` | Redis connection URL | `redis://localhost:6379` |
| `LOG_LEVEL` | Logging level | `INFO` |

### YAML Configuration

- `config/embedding_config.yaml` — Embedding model, versioning, drift detection
- `config/retrieval_config.yaml` — Hybrid weights, BM25 params, reranking
- `config/model_routing.yaml` — LLM selection based on query complexity
- `config/index_config.yaml` — Vector store and ANN index settings (HNSW/IVF)
- `config/reranker_config.yaml` — Cross-encoder and lightweight reranking config
- `config/logging.yaml` — Structured logging configuration

---

## 🧪 Testing

### Run Evaluation

```python
from monitoring.eval_runner import RAGEvaluator
from data.golden_eval_set import load_eval_set

evaluator = RAGEvaluator(pipeline=my_pipeline)
results = evaluator.run_evaluation(load_eval_set())

print(f"Recall@5: {results.mean_recall:.3f}")
print(f"P95 Latency: {results.p95_latency_ms:.1f}ms")
```

### Golden Dataset

```bash
# Run against golden evaluation set
python monitoring/eval_runner.py --eval-set data/golden_eval_set.json
```

---

## 🔒 Security

- **Row-level ACLs** via metadata filters
- **PII redaction** before indexing
- **Namespace isolation** for multi-tenancy
- **Audit logging** for compliance
- **Encrypted connections** (TLS)

---

## 📈 Monitoring

Export metrics to Prometheus/Grafana:

```bash
curl http://localhost:8000/metrics
```

Key metrics:

- `rag_query_latency_ms` — End-to-end latency
- `rag_retrieval_recall` — Recall@k
- `rag_cache_hit_rate` — Cache effectiveness
- `rag_hallucination_rate` — Verification failures

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m 'Add amazing feature'`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

Please ensure:

- Code is formatted with `black`
- Type hints are included
- Tests pass
- Documentation is updated

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Enterprise RAG Stack** — Built with ❤️ for production RAG systems

*Companion code for the Medium article series: [The Enterprise RAG Engineering Playbook](https://medium.com/@adnansattar09)*

[Report Bug](https://github.com/AdnanSattar/enterprise-rag-stack/issues) · [Request Feature](https://github.com/AdnanSattar/enterprise-rag-stack/issues)

</div>
