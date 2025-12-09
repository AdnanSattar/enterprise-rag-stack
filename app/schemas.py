"""
Pydantic schemas for API request/response models.
"""

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ConfidenceLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class QueryRequest(BaseModel):
    """Request model for RAG query endpoint."""

    query: str = Field(..., description="The user's question or search query")
    top_k: int = Field(
        default=5, ge=1, le=20, description="Number of results to return"
    )
    filters: Optional[Dict[str, Any]] = Field(
        default=None, description="Metadata filters for retrieval"
    )
    rerank: bool = Field(default=True, description="Whether to apply reranking")
    verify: bool = Field(default=True, description="Whether to verify the answer")
    prompt_type: str = Field(
        default="grounded_qa",
        description="Prompt template type: grounded_qa, structured, summarization",
    )


class SourceInfo(BaseModel):
    """Information about a source chunk."""

    doc_id: str
    chunk_id: str
    score: float
    title: Optional[str] = None


class QueryResponse(BaseModel):
    """Response model for RAG query endpoint."""

    answer: str
    sources: List[SourceInfo]
    confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM
    verified: bool = False
    verification_issues: List[str] = Field(default_factory=list)


class StructuredQueryResponse(BaseModel):
    """Structured response with JSON output from LLM."""

    answer: str
    sources: List[Dict[str, str]]
    confidence: ConfidenceLevel


class IngestRequest(BaseModel):
    """Request model for document ingestion."""

    file_path: str = Field(..., description="Path to file to ingest")
    tenant_id: str = Field(..., description="Tenant identifier")
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional metadata"
    )


class IngestBatchRequest(BaseModel):
    """Request model for batch ingestion."""

    directory: str = Field(..., description="Directory to ingest")
    tenant_id: str = Field(..., description="Tenant identifier")
    extensions: List[str] = Field(
        default=[".txt", ".md", ".pdf"], description="File extensions to process"
    )
    recursive: bool = Field(default=True, description="Search subdirectories")


class IngestResponse(BaseModel):
    """Response model for ingestion."""

    success: bool
    doc_id: Optional[str] = None
    chunks_created: int = 0
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class IngestBatchResponse(BaseModel):
    """Response model for batch ingestion."""

    total_files: int
    successful: int
    failed: int
    duplicates_removed: int
    total_chunks: int
    errors: List[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str
    vector_store_connected: bool
    document_count: int


class SearchRequest(BaseModel):
    """Request model for direct vector search (no LLM)."""

    query: str
    top_k: int = Field(default=10, ge=1, le=100)
    filters: Optional[Dict[str, Any]] = None
    include_vectors: bool = False


class SearchResult(BaseModel):
    """A single search result."""

    id: str
    doc_id: str
    text: str
    score: float
    metadata: Dict[str, Any]


class SearchResponse(BaseModel):
    """Response model for search endpoint."""

    results: List[SearchResult]
    total_found: int
    query_tokens: int


class DeleteRequest(BaseModel):
    """Request model for document deletion."""

    doc_ids: List[str] = Field(..., description="Document IDs to delete")
    tenant_id: str = Field(..., description="Tenant identifier for authorization")


class DeleteResponse(BaseModel):
    """Response model for deletion."""

    deleted_count: int
    errors: List[str] = Field(default_factory=list)
