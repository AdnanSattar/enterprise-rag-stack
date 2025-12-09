# Enterprise RAG Stack

> A production-grade Retrieval-Augmented Generation (RAG) implementation featuring hybrid search, semantic chunking, cross-encoder reranking, and multi-level answer verification.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive, battle-tested RAG service implementing the 15 best practices outlined in [RAG Engineering Best Practices](https://medium.com/) for building high-accuracy, low-latency, enterprise-grade AI systems.

## 🎯 Key Features

- **Hybrid Retrieval**: Vector + BM25 lexical search with score fusion
- **Semantic Chunking**: Token-aware chunking with overlap preservation
- **Embedding Versioning**: Full lifecycle management for embeddings
- **Reranking Pipeline**: Cross-encoder reranking for precision
- **Answer Verification**: Multi-level verification to prevent hallucinations
- **Circuit Breaker**: Resilient LLM calls with automatic recovery
- **Caching Layer**: Redis-backed caching for high-frequency queries
- **Multi-tenant Support**: Tenant isolation and metadata filtering
- **Production Observability**: Metrics, logging, and request tracing

## 📁 Project Structure

```
enterprise-rag-stack/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application
│   ├── config.py            # Configuration management
│   ├── schemas.py           # Pydantic request/response models
│   ├── ingestion.py         # Document ingestion pipeline
│   ├── chunking.py          # Semantic chunking
│   ├── embeddings.py        # Embedding generation with versioning
│   ├── retrieval.py         # Hybrid retrieval (vector + lexical)
│   ├── reranking.py         # Cross-encoder reranking
│   ├── verification.py      # Answer verification layer
│   ├── context_assembly.py  # Context building and prompts
│   ├── llm.py               # LLM client with model routing
│   ├── cache.py             # Caching layer (Redis/in-memory)
│   ├── circuit_breaker.py   # Resilience patterns
│   ├── evaluation.py        # Metrics and evaluation
│   └── graph_rag.py         # Knowledge graph integration
├── k8s/                     # Kubernetes manifests
├── notebooks/               # Demo notebooks
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

## 🚀 Quick Start

### Local Development

1. **Clone and install dependencies:**

```bash
git clone https://github.com/yourusername/enterprise-rag-stack.git
cd enterprise-rag-stack
pip install -r requirements.txt
```

2. **Set environment variables:**

```bash
export OPENAI_API_KEY="your-api-key"
export CHROMA_PATH="./data/chroma"
export COLLECTION_NAME="docs_v1"
```

3. **Run the service:**

```bash
uvicorn app.main:app --reload --port 8000
```

4. **Test the API:**

```bash
# Health check
curl http://localhost:8000/health

# Query endpoint
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the payment terms?", "top_k": 5}'
```

### Docker Deployment

```bash
# Start all services (Chroma, Redis, RAG API)
docker-compose up -d

# Check logs
docker-compose logs -f rag-api
```

### Kubernetes Deployment

```bash
# Create namespace and resources
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/secrets.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/pvc.yaml
kubectl apply -f k8s/chroma-deployment.yaml
kubectl apply -f k8s/redis-deployment.yaml
kubectl apply -f k8s/rag-deployment.yaml
kubectl apply -f k8s/ingress.yaml
```

## 📖 API Reference

### Query Endpoint

```http
POST /query
```

**Request:**

```json
{
  "query": "What are the payment terms?",
  "top_k": 5,
  "filters": {"tenant_id": "acme"},
  "rerank": true,
  "verify": true,
  "prompt_type": "grounded_qa"
}
```

**Response:**

```json
{
  "answer": "Payment must be made within 30 days...",
  "sources": [
    {"doc_id": "contract_001", "chunk_id": "chunk_0", "score": 0.92, "title": "Payment Terms"}
  ],
  "confidence": "high",
  "verified": true,
  "verification_issues": []
}
```

### Search Endpoint (No LLM)

```http
POST /search
```

### Document Ingestion

```http
POST /ingest
POST /ingest/batch
```

### Health & Metrics

```http
GET /health
GET /metrics
```

## 🏗️ Architecture

### The High-Accuracy RAG Formula

```
┌────────────────────────────────────────────────────────────────┐
│                  Enterprise RAG Stack Pipeline                 │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐ │
│   │  Ingest  │──▶│  Chunk   │ ─▶│  Embed    │──▶│  Index   │ │
│   │ & Clean  │    │ Semantic │    │ Version  │    │  HNSW    │ │
│   └──────────┘    └──────────┘    └──────────┘    └──────────┘ │
│                                                                │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐ │
│   │  Query   │──▶│  Hybrid  │──▶│ Rerank   │──▶│ Context│ │
│   │ Rewrite  │    │ Retrieve │    │ Top-K    │    │ Assemble │ │
│   └──────────┘    └──────────┘    └──────────┘    └──────────┘ │
│                                                                │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐                 │
│   │   LLM    │──▶│  Verify  │──▶│ Response │                │
│   │ Generate │    │  Answer  │    │  + Cache │                 │
│   └──────────┘    └──────────┘    └──────────┘                 │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### Key Components

| Component | Purpose | Key Practice |
|-----------|---------|--------------|
| **Ingestion** | Document normalization | Clean input beats clever retrieval |
| **Chunking** | Semantic segmentation | 150-300 words, 5-15% overlap |
| **Embedding** | Vector representation | Version with index, normalize vectors |
| **Retrieval** | Hybrid search | Vector + BM25 score fusion |
| **Reranking** | Precision boost | Cross-encoder on top-K candidates |
| **Verification** | Hallucination prevention | Deterministic + LLM critic |

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `CHROMA_PATH` | ChromaDB storage path | `/data/chroma` |
| `CHROMA_HOST` | Remote Chroma host (optional) | - |
| `COLLECTION_NAME` | Vector collection name | `docs_v1` |
| `OPENAI_API_KEY` | OpenAI API key | - |
| `REDIS_URL` | Redis connection URL | `redis://localhost:6379` |
| `CACHE_TTL` | Cache TTL in seconds | `3600` |
| `LOG_LEVEL` | Logging level | `INFO` |

### Embedding Configuration

```python
# app/config.py
@dataclass
class EmbeddingConfig:
    model_name: str = "all-MiniLM-L6-v2"
    normalize: bool = True
    version: str = "2025-01-01"  # Update on model changes!
```

**Critical Rule**: When changing `embedding_version`, you MUST re-index all documents.

## 📊 Best Practices Implemented

### 1. Document Ingestion

- ✅ Boilerplate removal
- ✅ OCR artifact cleaning
- ✅ Content deduplication
- ✅ Metadata extraction

### 2. Chunking Strategy

- ✅ Semantic boundaries (paragraphs, headings)
- ✅ Token-aware limits (800-1500 tokens)
- ✅ Overlap preservation (5-15%)
- ✅ Structure preservation

### 3. Embedding Management

- ✅ Version tracking with index
- ✅ Unit vector normalization
- ✅ Drift detection monitoring
- ✅ Deterministic preprocessing

### 4. Hybrid Retrieval

- ✅ Vector similarity (semantic)
- ✅ BM25 scoring (lexical)
- ✅ Score normalization and fusion
- ✅ Metadata filtering

### 5. Reranking Pipeline

- ✅ Cross-encoder reranking
- ✅ Chunk deduplication
- ✅ Score blending

### 6. Verification Layer

- ✅ Deterministic checks (numbers, dates)
- ✅ LLM critic validation
- ✅ Fallback responses

### 7. Operational Excellence

- ✅ Circuit breaker for LLM calls
- ✅ Request-level caching
- ✅ Model routing by complexity
- ✅ Health checks and metrics

## 🧪 Testing

### Run the Demo Notebook

```bash
cd notebooks
jupyter notebook rag_demo.ipynb
```

### Evaluation Framework

```python
from app.evaluation import RAGEvaluator, EvalItem

# Define evaluation set
eval_set = [
    EvalItem(
        query="What are the payment terms?",
        expected_doc_ids=["contract_001#chunk_0"]
    ),
    # ... more items
]

# Run evaluation
evaluator = RAGEvaluator(pipeline=my_pipeline)
results = evaluator.run_evaluation(eval_set)

print(f"Mean Recall: {results.mean_recall:.3f}")
print(f"Mean Precision: {results.mean_precision:.3f}")
print(f"P95 Latency: {results.p95_latency_ms:.1f}ms")
```

## 📈 Monitoring

### Key Metrics to Track

| Metric | Target | Alert Threshold |
|--------|--------|-----------------|
| Recall@K | >0.85 | <0.75 |
| Precision@K | >0.70 | <0.60 |
| P95 Latency | <2000ms | >5000ms |
| Hallucination Rate | <5% | >10% |
| Cache Hit Rate | >40% | <20% |

### Prometheus Metrics Endpoint

```bash
curl http://localhost:8000/metrics
```

## 🔒 Security

- Row-level ACLs via metadata filters
- Tenant namespace isolation
- PII detection (extensible)
- Audit logging for all queries
- Encrypted connections (TLS)

## 📚 References

- [Pinecone: RAG Best Practices](https://www.pinecone.io/)
- [OpenSearch: Hybrid Search](https://opensearch.org/)
- [Chunking Strategies](https://medium.com/)
- [Embedding Drift Detection](https://www.evidentlyai.com/)
- [ANN Index Tradeoffs (HNSW vs IVF)](https://www.tidb.io/)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.

---

**Enterprise RAG Stack** — Built with ❤️ for production RAG systems

*Companion code for the Medium article: [RAG Engineering Best Practices](https://medium.com/)*
