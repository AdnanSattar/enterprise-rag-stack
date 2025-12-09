"""
Main FastAPI application for Enterprise RAG Stack.

Production-ready API with:
- Async endpoints
- Circuit breaker protection
- Caching
- Health checks
- Request tracing
"""

import logging
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from chunking.semantic_chunker import chunk_document
from context.context_builder import assemble_context, build_prompt
from cost_optimization.caching_layer import get_cache, make_cache_key
from cost_optimization.model_router import get_model_router
from deployment.circuit_breaker import CircuitOpenError
from ingestion.ingest_pipeline import IngestionPipeline
from monitoring.eval_runner import get_metrics
from reranking.cross_encoder_reranker import rerank_results
from retrieval.hybrid_retriever import get_retriever, get_vector_store
from verification.verification_pipeline import VerificationStatus, verify_answer

# Import schemas from shared module
try:
    from shared.schemas import (
        ConfidenceLevel,
        DeleteRequest,
        DeleteResponse,
        HealthResponse,
        IngestBatchRequest,
        IngestBatchResponse,
        IngestRequest,
        IngestResponse,
        QueryRequest,
        QueryResponse,
        SearchRequest,
        SearchResponse,
        SearchResult,
        SourceInfo,
    )
except ImportError:
    # Fallback if schemas moved
    from enum import Enum
    from typing import Any, Dict, List, Optional

    from pydantic import BaseModel

    class ConfidenceLevel(str, Enum):
        LOW = "low"
        MEDIUM = "medium"
        HIGH = "high"

    # Minimal schemas - full ones should be in a shared location
    class QueryRequest(BaseModel):
        query: str
        top_k: int = 5
        filters: Optional[Dict[str, Any]] = None
        rerank: bool = True
        verify: bool = True
        prompt_type: str = "grounded_qa"

    class SourceInfo(BaseModel):
        doc_id: str
        chunk_id: str
        score: float
        title: Optional[str] = None

    class QueryResponse(BaseModel):
        answer: str
        sources: List[SourceInfo]
        confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM
        verified: bool = False
        verification_issues: List[str] = []

    class HealthResponse(BaseModel):
        status: str
        version: str
        vector_store_connected: bool
        document_count: int

    class SearchRequest(BaseModel):
        query: str
        top_k: int = 10
        filters: Optional[Dict[str, Any]] = None

    class SearchResponse(BaseModel):
        results: List[SearchResult]
        total_found: int
        query_tokens: int

    class SearchResult(BaseModel):
        id: str
        doc_id: str
        text: str
        score: float
        metadata: Dict[str, Any]

    class IngestRequest(BaseModel):
        file_path: str
        tenant_id: str
        metadata: Optional[Dict[str, Any]] = None

    class IngestResponse(BaseModel):
        success: bool
        doc_id: Optional[str] = None
        chunks_created: int = 0
        warnings: List[str] = []
        errors: List[str] = []

    class IngestBatchRequest(BaseModel):
        directory: str
        tenant_id: str
        extensions: List[str] = [".txt", ".pdf", ".md"]
        recursive: bool = True

    class IngestBatchResponse(BaseModel):
        total_files: int
        successful: int
        failed: int
        duplicates_removed: int
        total_chunks: int
        errors: List[str] = []

    class DeleteRequest(BaseModel):
        doc_ids: List[str]

    class DeleteResponse(BaseModel):
        deleted_count: int
        errors: List[str] = []


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

__version__ = "1.0.0"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    logger.info(f"Starting Enterprise RAG Stack v{__version__}")

    # Initialize vector store connection
    try:
        store = get_vector_store()
        count = store.count()
        logger.info(f"Connected to vector store. Document count: {count}")
    except Exception as e:
        logger.warning(f"Vector store connection failed: {e}")

    yield

    # Shutdown
    logger.info("Shutting down Enterprise RAG Stack")


app = FastAPI(
    title="Enterprise RAG Stack",
    description="Production-grade Retrieval-Augmented Generation API",
    version=__version__,
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Add request ID for tracing."""
    request_id = str(uuid.uuid4())[:8]
    request.state.request_id = request_id

    start_time = time.time()
    response = await call_next(request)

    # Log request
    duration_ms = (time.time() - start_time) * 1000
    logger.info(
        f"[{request_id}] {request.method} {request.url.path} "
        f"- {response.status_code} - {duration_ms:.1f}ms"
    )

    response.headers["X-Request-ID"] = request_id
    return response


@app.exception_handler(CircuitOpenError)
async def circuit_open_handler(request: Request, exc: CircuitOpenError):
    """Handle circuit breaker open."""
    return JSONResponse(
        status_code=503,
        content={
            "error": "Service temporarily unavailable",
            "detail": str(exc),
            "retry_after": 30,
        },
        headers={"Retry-After": "30"},
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        store = get_vector_store()
        doc_count = store.count()
        connected = True
    except Exception:
        doc_count = 0
        connected = False

    return HealthResponse(
        status="healthy" if connected else "degraded",
        version=__version__,
        vector_store_connected=connected,
        document_count=doc_count,
    )


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(body: QueryRequest, request: Request):
    """
    Main RAG query endpoint.

    Flow:
    1. Check cache
    2. Hybrid retrieval
    3. Rerank
    4. Assemble context
    5. Generate answer
    6. Verify
    7. Cache result
    """
    request_id = getattr(request.state, "request_id", "unknown")
    metrics = get_metrics()
    start_time = time.time()

    # Check cache
    cache = get_cache()
    cache_key = make_cache_key(body.query, body.top_k, body.filters)

    cached = cache.get(cache_key)
    if cached:
        metrics.record("cache_hits", 1)
        logger.info(f"[{request_id}] Cache hit")
        return QueryResponse(**cached)

    try:
        # 1. Hybrid retrieval
        retrieval_start = time.time()
        retriever = get_retriever()
        candidates = retriever.retrieve(
            query=body.query,
            top_k=body.top_k * 2 if body.rerank else body.top_k,
            filters=body.filters,
        )
        metrics.record("retrieval_latency_ms", (time.time() - retrieval_start) * 1000)
        metrics.record("chunks_retrieved", len(candidates))

        if not candidates:
            return QueryResponse(
                answer="Not found in context",
                sources=[],
                confidence=ConfidenceLevel.LOW,
                verified=False,
            )

        # 2. Rerank
        if body.rerank:
            reranked = rerank_results(
                query=body.query,
                candidates=candidates,
                top_k=body.top_k,
                deduplicate=True,
                use_lightweight=True,
            )
        else:
            from reranking.cross_encoder_reranker import RerankResult

            reranked = [
                RerankResult(
                    id=c.id,
                    doc_id=c.doc_id,
                    text=c.text,
                    metadata=c.metadata,
                    original_score=c.combined_score,
                    rerank_score=c.combined_score,
                    final_score=c.combined_score,
                )
                for c in candidates[: body.top_k]
            ]

        # 3. Assemble context
        assembled = assemble_context(reranked)

        # 4. Build prompt and generate
        prompt = build_prompt(
            context=assembled.text, query=body.query, prompt_type=body.prompt_type
        )

        llm_start = time.time()
        router = get_model_router()
        response = router.generate(prompt=prompt, query=body.query)
        metrics.record("llm_latency_ms", (time.time() - llm_start) * 1000)
        metrics.record("tokens_used", response.usage.get("total_tokens", 0))

        answer = response.content

        # 5. Verify answer
        verified = False
        verification_issues = []
        confidence = ConfidenceLevel.MEDIUM

        if body.verify:
            verification = verify_answer(answer, assembled.text)
            verified = verification.status == VerificationStatus.VALID
            verification_issues = verification.issues

            if verified:
                confidence = ConfidenceLevel.HIGH
            elif verification.status == VerificationStatus.INVALID:
                confidence = ConfidenceLevel.LOW

        # Build response
        sources = [
            SourceInfo(
                doc_id=s["doc_id"],
                chunk_id=s["chunk_id"],
                score=s["score"],
                title=s.get("title"),
            )
            for s in assembled.sources
        ]

        result = QueryResponse(
            answer=answer,
            sources=sources,
            confidence=confidence,
            verified=verified,
            verification_issues=verification_issues,
        )

        # Cache result
        cache.set(cache_key, result.model_dump(), ttl=3600)

        metrics.record("query_latency_ms", (time.time() - start_time) * 1000)

        return result

    except CircuitOpenError:
        raise
    except Exception as e:
        logger.exception(f"[{request_id}] Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search", response_model=SearchResponse)
async def search_endpoint(body: SearchRequest):
    """
    Direct vector search without LLM generation.
    Useful for debugging and exploring the index.
    """
    from chunking.semantic_chunker import num_tokens

    retriever = get_retriever()
    results = retriever.retrieve(
        query=body.query, top_k=body.top_k, filters=body.filters
    )

    return SearchResponse(
        results=[
            SearchResult(
                id=r.id,
                doc_id=r.doc_id,
                text=r.text,
                score=r.combined_score,
                metadata=r.metadata,
            )
            for r in results
        ],
        total_found=len(results),
        query_tokens=num_tokens(body.query),
    )


@app.post("/ingest", response_model=IngestResponse)
async def ingest_endpoint(body: IngestRequest):
    """Ingest a single document."""
    try:
        pipeline = IngestionPipeline(tenant_id=body.tenant_id)
        result = pipeline.process_file(body.file_path, body.metadata)

        if result is None:
            return IngestResponse(
                success=False, errors=["Failed to process file or duplicate detected"]
            )

        # Chunk the document
        chunks = chunk_document(
            text=result.text, doc_id=result.doc_id, metadata=result.metadata
        )

        # Add to vector store
        store = get_vector_store()
        store.add_documents(
            ids=[c["id"] for c in chunks],
            texts=[c["text"] for c in chunks],
            metadatas=[
                {k: v for k, v in c.items() if k not in ["id", "text"]} for c in chunks
            ],
        )

        return IngestResponse(
            success=True,
            doc_id=result.doc_id,
            chunks_created=len(chunks),
            warnings=result.warnings,
        )

    except Exception as e:
        logger.exception(f"Ingestion failed: {e}")
        return IngestResponse(success=False, errors=[str(e)])


@app.post("/ingest/batch", response_model=IngestBatchResponse)
async def ingest_batch_endpoint(body: IngestBatchRequest):
    """Batch ingest documents from a directory."""
    try:
        pipeline = IngestionPipeline(tenant_id=body.tenant_id)
        results = pipeline.process_directory(
            directory=body.directory,
            extensions=body.extensions,
            recursive=body.recursive,
        )

        total_chunks = 0
        errors = []
        store = get_vector_store()

        for result in results:
            try:
                chunks = chunk_document(
                    text=result.text, doc_id=result.doc_id, metadata=result.metadata
                )

                store.add_documents(
                    ids=[c["id"] for c in chunks],
                    texts=[c["text"] for c in chunks],
                    metadatas=[
                        {k: v for k, v in c.items() if k not in ["id", "text"]}
                        for c in chunks
                    ],
                )

                total_chunks += len(chunks)

            except Exception as e:
                errors.append(
                    f"Error processing {result.metadata.get('source_path')}: {e}"
                )

        stats = pipeline.get_stats()

        return IngestBatchResponse(
            total_files=stats["total_docs"],
            successful=stats["successful"],
            failed=stats["failed"],
            duplicates_removed=stats["duplicates_removed"],
            total_chunks=total_chunks,
            errors=errors,
        )

    except Exception as e:
        logger.exception(f"Batch ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/documents", response_model=DeleteResponse)
async def delete_documents(body: DeleteRequest):
    """Delete documents by ID."""
    try:
        store = get_vector_store()

        deleted_count = 0
        errors = []

        for doc_id in body.doc_ids:
            try:
                # Delete chunks with this doc_id prefix
                store.delete([f"{doc_id}#chunk_{i}" for i in range(100)])
                deleted_count += 1
            except Exception as e:
                errors.append(f"Error deleting {doc_id}: {e}")

        return DeleteResponse(deleted_count=deleted_count, errors=errors)

    except Exception as e:
        logger.exception(f"Delete failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def metrics_endpoint():
    """Get operational metrics."""
    metrics = get_metrics()
    return metrics.get_summary()


@app.post("/metrics/reset")
async def reset_metrics():
    """Reset operational metrics."""
    metrics = get_metrics()
    metrics.reset()
    return {"status": "reset"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
