"""
Configuration module for RAG Service.
Manages all environment variables and settings with validation.
"""

import os
from dataclasses import dataclass, field
from typing import Optional
from functools import lru_cache


@dataclass
class EmbeddingConfig:
    """Embedding model configuration - version this with your index."""
    model_name: str = "all-MiniLM-L6-v2"
    normalize: bool = True
    version: str = "2025-01-01"
    dimension: int = 384  # Matches all-MiniLM-L6-v2


@dataclass
class ChunkingConfig:
    """Chunking strategy configuration."""
    max_tokens: int = 1200
    overlap_tokens: int = 150
    min_chunk_size: int = 100


@dataclass
class RetrievalConfig:
    """Retrieval and reranking configuration."""
    default_top_k: int = 10
    rerank_top_k: int = 5
    vector_weight: float = 0.7
    lexical_weight: float = 0.3
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@dataclass
class LLMConfig:
    """LLM configuration for generation."""
    model: str = "gpt-4.1-mini"
    temperature: float = 0.0
    max_tokens: int = 2048
    timeout: int = 30


@dataclass
class Settings:
    """Main application settings loaded from environment."""
    
    # Chroma settings
    CHROMA_PATH: str = field(default_factory=lambda: os.getenv("CHROMA_PATH", "/data/chroma"))
    CHROMA_HOST: Optional[str] = field(default_factory=lambda: os.getenv("CHROMA_HOST"))
    CHROMA_PORT: int = field(default_factory=lambda: int(os.getenv("CHROMA_PORT", "8000")))
    COLLECTION_NAME: str = field(default_factory=lambda: os.getenv("COLLECTION_NAME", "docs_v1"))
    
    # OpenAI settings
    OPENAI_API_KEY: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    
    # Redis cache settings
    REDIS_URL: str = field(default_factory=lambda: os.getenv("REDIS_URL", "redis://localhost:6379"))
    CACHE_TTL: int = field(default_factory=lambda: int(os.getenv("CACHE_TTL", "3600")))
    
    # Application settings
    DEBUG: bool = field(default_factory=lambda: os.getenv("DEBUG", "false").lower() == "true")
    LOG_LEVEL: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    
    # Nested configs
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    
    def get_index_name(self) -> str:
        """Generate versioned index name for embedding schema management."""
        model_slug = self.embedding.model_name.replace("/", "_").replace("-", "_")
        return f"{self.COLLECTION_NAME}_{model_slug}_{self.embedding.version}"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


settings = get_settings()

