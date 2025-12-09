# Enterprise RAG Engineering Playbook - Article Series Index

This repository accompanies the article series **"The Enterprise RAG Engineering Playbook"** published on Medium.

---

## 📖 Article Series Overview

A comprehensive guide to building production-grade RAG systems, covering everything from document ingestion to deployment and monitoring.

---

## Part 1: High Recall Retrieval and Precision Reranking

**🔗 [Read on Medium](https://medium.com/@adnansattar09/rag-engineering-part-2-high-recall-retrieval-and-precision-reranking)**

**Topics Covered:**

- Vector stores and ANN indexing (HNSW, IVF)
- Hybrid retrieval: Vector + BM25 score fusion
- Cross-encoder reranking pipeline
- Chunk deduplication strategies

**Related Code:**

- `retrieval/` - Hybrid retrieval implementation
- `reranking/` - Cross-encoder reranking
- `config/retrieval_config.yaml` - Retrieval configuration
- `config/reranker_config.yaml` - Reranker settings

**Key Takeaways:**

- Pure vector search fails for keyword-heavy queries
- Score fusion: `score = w_vec × s_vec + w_bm25 × s_bm25`
- Rerank only top 20-50 candidates for efficiency

---

## Part 2: Context Assembly and Grounded Prompting

**🔗 [Read on Medium](https://medium.com/@adnansattar09/rag-engineering-part-3-context-assembly-and-grounded-prompting)**

**Topics Covered:**

- Token-budgeted context assembly
- Structured prompt templates
- Inline citation formatting
- Grounded LLM prompting to prevent hallucinations

**Related Code:**

- `context/context_builder.py` - Context assembly logic
- `context/context_budgeting.py` - Token budgeting
- `context/prompt_templates/` - Prompt templates (compliance, summarization, grounding)

**Key Takeaways:**

- Choose between "few long chunks" vs "many small chunks"
- Add lightweight headings like `[Section: Payment Terms]`
- Format citations for downstream UI mapping

---

## Part 3: Context Assembly and Grounded Prompting

**🔗 [Read on Medium](https://medium.com/@adnansattar09/rag-engineering-part-3-context-assembly-and-grounded-prompting)**

*Note: This appears to be a duplicate of Part 2. Please verify the correct link.*

---

## Part 4: Verification Layers and Graph-RAG

**🔗 [Read on Medium](https://medium.com/@adnansattar09/rag-engineering-part-4-verification-layers-and-graph-rag-for-trustworthy-answers)**

**Topics Covered:**

- Two-level verification: deterministic checks + LLM critic
- Graph-RAG for multi-hop queries
- Entity linking and knowledge graph traversal
- Subgraph-to-text conversion

**Related Code:**

- `verification/` - Answer verification pipeline
- `graph_rag/` - Knowledge graph integration
- `graph_rag/entity_linking.py` - Entity extraction
- `graph_rag/graph_traversal.py` - Multi-hop traversal

**Key Takeaways:**

- Level 1: Regex checks for numbers, dates, allowed values
- Level 2: Critic LLM validates factual alignment
- Graph-RAG enables multi-hop queries like "Which products are impacted by delayed contracts in APAC"

---

## Part 5: Monitoring, Evaluation and Lifecycle Management

**🔗 [Read on Medium](https://medium.com/@adnansattar09/rag-engineering-part-5-monitoring-evaluation-and-lifecycle-management)**

**Topics Covered:**

- Recall@K and precision metrics
- Latency tracking (P50, P95, P99)
- Embedding drift detection
- Golden dataset evaluation
- Shadow evaluation for A/B testing

**Related Code:**

- `monitoring/eval_runner.py` - Evaluation framework
- `monitoring/recall_metrics.py` - Recall, Precision, MRR, NDCG
- `monitoring/latency_metrics.py` - Latency tracking
- `monitoring/evaluation_schema.md` - Evaluation data schemas
- `deployment/` - FastAPI, Docker, Kubernetes configs

**Key Takeaways:**

- Track task-level metrics (answer correctness) and system-level metrics (latency, cost)
- Use shadow evaluation to test pipeline changes offline
- Monitor embedding drift to detect model degradation

---

## Part 6: Security, Compliance and Cost Optimization

**🔗 [Read on Medium](https://medium.com/@adnansattar09/rag-engineering-part-6-security-compliance-and-cost-optimization)**

**Topics Covered:**

- Row-level ACLs for multi-tenant isolation
- PII redaction before indexing
- Audit logging for compliance
- Model routing for cost optimization
- Response caching strategies
- GPU batching for embeddings

**Related Code:**

- `security/pii_redaction.py` - PII detection and redaction
- `security/acl_filters.py` - Multi-tenant ACL filtering
- `security/audit_logging.py` - Compliance audit logs
- `cost_optimization/model_router.py` - Model routing by complexity
- `cost_optimization/caching_layer.py` - Redis-backed caching
- `cost_optimization/gpu_batcher.py` - GPU batching
- `cost_optimization/unit_economics_calculator.py` - Cost tracking

**Key Takeaways:**

- Use metadata filters for tenant isolation
- Redact PII before indexing to protect sensitive data
- Route simple queries to smaller models to reduce costs
- Cache high-frequency queries to improve latency

---

## 🏗️ Repository Structure

Each article maps to specific modules in this repository:

```
enterprise-rag-stack/
├── ingestion/          # Document normalization (foundation)
├── chunking/           # Semantic chunking (foundation)
├── embeddings/         # Embedding versioning (foundation)
├── retrieval/          # Part 1: Hybrid retrieval
├── reranking/          # Part 1: Cross-encoder reranking
├── context/            # Part 2: Context assembly & prompts
├── verification/       # Part 4: Answer verification
├── graph_rag/          # Part 4: Knowledge graph integration
├── monitoring/         # Part 5: Evaluation & metrics
├── deployment/         # Part 5: FastAPI, Docker, K8s
├── security/           # Part 6: ACLs, PII, audit
└── cost_optimization/ # Part 6: Model routing, caching
```

---

## 🚀 Getting Started

1. **Read the articles** to understand the concepts
2. **Explore the code** in the corresponding modules
3. **Run the examples** in `notebooks/rag_demo.ipynb`
4. **Deploy** using Docker or Kubernetes configs

---

## 📝 Additional Resources

- **Architecture Overview**: [docs/architecture_overview.md](architecture_overview.md)
- **Configuration Guide**: See `config/` directory
- **Sample Data**: See `data/` directory
- **Evaluation Schema**: [monitoring/evaluation_schema.md](../monitoring/evaluation_schema.md)

---

## 🤝 Contributing

Found an issue or want to improve the code? Contributions are welcome!

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

**Built with ❤️ for production RAG systems**
