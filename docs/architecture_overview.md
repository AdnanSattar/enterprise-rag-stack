# Enterprise RAG Stack Architecture

## Overview

The Enterprise RAG Stack implements a complete Retrieval-Augmented Generation pipeline for production environments. This document describes the system architecture, data flow, and key design decisions.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Enterprise RAG Stack                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐          │
│  │  Ingest  │───▶│  Chunk   │───▶│  Embed   │───▶│  Index   │          │
│  │ & Clean  │    │ Semantic │    │ Version  │    │  HNSW    │          │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘          │
│                                                                          │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐          │
│  │  Query   │───▶│  Hybrid  │───▶│ Rerank   │───▶│ Context  │          │
│  │ Classify │    │ Retrieve │    │ Top-K    │    │ Assemble │          │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘          │
│                                                                          │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐                           │
│  │   LLM    │───▶│  Verify  │───▶│ Response │                          │
│  │ Generate │    │  Answer  │    │  + Cache │                           │
│  └──────────┘    └──────────┘    └──────────┘                           │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Document Ingestion (`ingestion/`)

**Purpose**: Clean input beats clever retrieval.

**Pipeline**:

```
Raw files → Parsers (PDF, HTML, Office) → Cleaning & dedupe → Normalized text + metadata
```

**Key Features**:

- Multi-format support via `unstructured` library
- Boilerplate removal (headers, footers, disclaimers)
- OCR artifact cleaning
- Content-based deduplication
- Metadata extraction

### 2. Semantic Chunking (`chunking/`)

**Purpose**: Chunking determines what the retriever can find.

**Strategies**:

- Title-aware chunking (respects document structure)
- Section-aware chunking (maintains hierarchy)
- Sentence segmentation (for fine-grained control)

**Guidelines**:

| Domain | Chunk Size | Overlap | Rationale |
|--------|------------|---------|-----------|
| Legal | 800-1200 tokens | 15% | Complete clauses |
| Code | 400-800 tokens | 10% | Function boundaries |
| FAQs | 200-400 tokens | 5% | Discrete answers |
| General | 600-1000 tokens | 10% | Balanced |

### 3. Embeddings (`embeddings/`)

**Critical Rule**: Never mix vectors from different models in the same index.

**Versioning Pattern**:

```python
index_name = f"docs_v1_{model_slug}_{embedding_version}"
```

**Features**:

- Deterministic preprocessing
- Unit vector normalization
- Drift detection and alerting

### 4. Hybrid Retrieval (`retrieval/`)

**Formula**:

```
score = w_vec × s_vec + w_bm25 × s_bm25
```

**Why Hybrid?**:

- Pure vector fails for keyword-heavy queries (SKU numbers, codes)
- BM25 captures exact term matches
- Score fusion balances semantic + lexical

**Query Classification**:

| Query Type | Vector Weight | Lexical Weight |
|------------|---------------|----------------|
| Factual | 0.5 | 0.5 |
| Exploratory | 0.8 | 0.2 |
| Navigational | 0.3 | 0.7 |

### 5. Reranking (`reranking/`)

**Purpose**: Cost-effective accuracy boost.

**Implementation**:

- Cross-encoder for precision (top 20-50 candidates only)
- Lightweight embedding reranker for speed
- Chunk deduplication (prevent redundant context)

**Important**: Log original scores alongside reranker scores for analysis.

### 6. Context Assembly (`context/`)

**Budget Management**:

```
Available = Model_Max - System_Prompt - Query - Response_Reserve
```

**Citation Format**:

```
[Source 1] doc=contract_001 chunk=chunk_0 | score: 0.92
Title: Payment Terms

Payment is due within 30 days of the invoice date...
```

### 7. Verification (`verification/`)

**Two-Level Verification**:

| Level | Method | Checks |
|-------|--------|--------|
| 1 | Deterministic | Numbers, dates, currencies in context |
| 2 | LLM Critic | Factual alignment, unsupported claims |

### 8. Graph-RAG (`graph_rag/`)

**For Multi-Hop Queries**:

```
"Which products are impacted by delayed contracts in APAC?"

1. Find contracts: status = "Delayed"
2. Filter by region: APAC (via customer relationship)
3. Follow relations: Contract → PERTAINS_TO → Product
```

## Data Flow

### Indexing Flow

```
1. Document Upload
2. Normalization (clean, dedupe)
3. Semantic Chunking
4. Embedding Generation
5. Vector Store Indexing
6. Metadata Persistence
```

### Query Flow

```
1. Query Reception
2. Cache Check
3. Query Classification
4. Hybrid Retrieval (Vector + BM25)
5. Reranking (Cross-encoder)
6. Context Assembly
7. LLM Generation
8. Answer Verification
9. Response + Cache
```

## Deployment Architecture

### Single Node (Development)

```
┌─────────────────────────┐
│    Docker Compose       │
├─────────────────────────┤
│  RAG Service (FastAPI)  │
│  ChromaDB (embedded)    │
│  Redis (cache)          │
└─────────────────────────┘
```

### Kubernetes (Production)

```
┌─────────────────────────────────────────┐
│            Kubernetes Cluster            │
├─────────────────────────────────────────┤
│  ┌─────────┐  ┌─────────┐  ┌─────────┐ │
│  │ RAG API │  │ RAG API │  │ RAG API │ │
│  │ Pod 1   │  │ Pod 2   │  │ Pod N   │ │
│  └────┬────┘  └────┬────┘  └────┬────┘ │
│       │            │            │       │
│  ┌────┴────────────┴────────────┴────┐ │
│  │         Load Balancer             │ │
│  └───────────────────────────────────┘ │
│                                         │
│  ┌─────────────┐  ┌─────────────────┐  │
│  │  ChromaDB   │  │      Redis      │  │
│  │  (PVC)      │  │    (Cluster)    │  │
│  └─────────────┘  └─────────────────┘  │
└─────────────────────────────────────────┘
```

## Key Design Decisions

1. **ChromaDB over Pinecone**: In-process for simplicity, HNSW for recall
2. **Hybrid over Pure Vector**: Better for keyword-heavy enterprise queries
3. **Cross-encoder Reranking**: Worth the latency for accuracy-critical apps
4. **Two-level Verification**: Deterministic first, LLM critic for edge cases
5. **Circuit Breaker**: Protect against LLM provider failures

## Performance Targets

| Metric | Target | Alert Threshold |
|--------|--------|-----------------|
| Recall@5 | > 0.85 | < 0.75 |
| Precision@5 | > 0.70 | < 0.60 |
| P95 Latency | < 2000ms | > 5000ms |
| Hallucination Rate | < 5% | > 10% |
| Cache Hit Rate | > 40% | < 20% |
