"""
Vector retrieval using ChromaDB.

Supports:
- HNSW index (high recall, update-friendly)
- Metadata filtering at query time
- Embedding versioning
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import chromadb
import numpy as np
from chromadb.config import Settings as ChromaSettings

logger = logging.getLogger(__name__)


@dataclass
class VectorSearchResult:
    """Result from vector search."""
    
    id: str
    text: str
    metadata: Dict
    distance: float
    score: float  # Converted similarity score


class VectorStore:
    """
    Vector store wrapper with Chroma backend.
    
    Supports:
    - HNSW index (high recall, update-friendly)
    - Metadata filtering at query time
    - Embedding versioning
    
    Usage:
        store = VectorStore(path="./chroma_data", collection_name="docs_v1")
        store.add_documents(ids, texts, metadatas)
        results = store.query("search query", n_results=10)
    """
    
    def __init__(
        self,
        path: str = None,
        collection_name: str = None,
        host: str = None,
        port: int = None,
    ):
        """
        Args:
            path: Local storage path
            collection_name: Collection name
            host: Remote Chroma host (optional)
            port: Remote Chroma port
        """
        self.path = path or "./data/chroma"
        self.collection_name = collection_name or "docs_v1"
        self.host = host
        self.port = port or 8000
        
        self._client: Optional[chromadb.ClientAPI] = None
        self._collection = None
    
    @property
    def client(self) -> chromadb.ClientAPI:
        """Get or create Chroma client."""
        if self._client is None:
            if self.host:
                # Remote client
                self._client = chromadb.HttpClient(
                    host=self.host,
                    port=self.port
                )
            else:
                # Local persistent client
                self._client = chromadb.PersistentClient(
                    path=self.path,
                    settings=ChromaSettings(anonymized_telemetry=False)
                )
        return self._client
    
    @property
    def collection(self):
        """Get or create collection."""
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection
    
    def add_documents(
        self,
        ids: List[str],
        texts: List[str],
        metadatas: List[Dict],
        embeddings: Optional[np.ndarray] = None,
    ) -> None:
        """
        Add documents to the vector store.
        
        Args:
            ids: Unique identifiers
            texts: Document texts
            metadatas: Metadata for each document
            embeddings: Pre-computed embeddings (computed if None)
        """
        if embeddings is None:
            from embeddings import embed_batch
            embeddings, config = embed_batch(texts)
            # Add embedding metadata
            for meta in metadatas:
                meta["embedding_model"] = config.model_name
                meta["embedding_version"] = config.version
        
        self.collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings.tolist() if isinstance(embeddings, np.ndarray) else embeddings,
        )
        
        logger.info(f"Added {len(ids)} documents to collection {self.collection_name}")
    
    def query(
        self,
        query_text: str = None,
        query_embedding: np.ndarray = None,
        n_results: int = 10,
        where: Optional[Dict] = None,
        where_document: Optional[Dict] = None,
    ) -> Dict:
        """
        Query the vector store.
        
        Args:
            query_text: Query string (embedding computed automatically)
            query_embedding: Pre-computed query embedding
            n_results: Number of results to return
            where: Metadata filter
            where_document: Document content filter
            
        Returns:
            Chroma query results dict
        """
        if query_embedding is None:
            if query_text is None:
                raise ValueError("Either query_text or query_embedding required")
            from embeddings import embed_batch
            query_embedding, _ = embed_batch([query_text])
        
        return self.collection.query(
            query_embeddings=query_embedding.tolist() if isinstance(query_embedding, np.ndarray) else query_embedding,
            n_results=n_results,
            where=where,
            where_document=where_document,
            include=["documents", "metadatas", "distances"],
        )
    
    def delete(self, ids: List[str]) -> None:
        """Delete documents by ID."""
        self.collection.delete(ids=ids)
        logger.info(f"Deleted {len(ids)} documents")
    
    def count(self) -> int:
        """Get document count."""
        return self.collection.count()
    
    def get_by_ids(self, ids: List[str]) -> Dict:
        """Get documents by ID."""
        return self.collection.get(ids=ids, include=["documents", "metadatas"])


class VectorRetriever:
    """
    Pure vector retrieval.
    
    Use HybridRetriever for production - pure vector fails
    for keyword-heavy queries (SKU numbers, technical codes).
    """
    
    def __init__(self, vector_store: VectorStore = None):
        self.vector_store = vector_store or VectorStore()
    
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict] = None,
    ) -> List[VectorSearchResult]:
        """
        Retrieve documents using vector similarity.
        
        Args:
            query: Search query
            top_k: Number of results
            filters: Metadata filters
            
        Returns:
            List of VectorSearchResult
        """
        results = self.vector_store.query(
            query_text=query,
            n_results=top_k,
            where=filters,
        )
        
        if not results["ids"][0]:
            return []
        
        output = []
        for i in range(len(results["ids"][0])):
            distance = results["distances"][0][i]
            # Convert distance to similarity (Chroma returns L2 distance)
            score = 1 - (distance / 2)
            
            output.append(VectorSearchResult(
                id=results["ids"][0][i],
                text=results["documents"][0][i],
                metadata=results["metadatas"][0][i],
                distance=distance,
                score=score,
            ))
        
        return output

