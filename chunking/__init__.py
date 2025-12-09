"""
Semantic Chunking Module.

Chunking determines what the retriever can find.
This module provides multiple chunking strategies:
- Semantic chunking (paragraph/section-aware)
- Sentence-based splitting
- Token-budget-based segmentation

Rules of thumb:
- Prefer semantic chunking over fixed length
- Aim for 150-300 words or 800-1500 tokens
- Use 5-15% overlap to preserve context
- Preserve hierarchical structure

Usage:
    from chunking import SemanticChunker, chunk_document

    chunker = SemanticChunker(max_tokens=1000, overlap_tokens=100)
    chunks = chunker.chunk(text)
"""

from .chunk_eval_tools import (
    ChunkQualityReport,
    compute_chunk_coherence,
    evaluate_chunk_quality,
)
from .semantic_chunker import (
    Chunk,
    Paragraph,
    SemanticChunker,
    chunk_document,
    extract_paragraphs,
    semantic_chunk,
)
from .sentence_splitter import SentenceSplitter, split_into_sentences

__all__ = [
    "SemanticChunker",
    "Paragraph",
    "Chunk",
    "extract_paragraphs",
    "semantic_chunk",
    "chunk_document",
    "SentenceSplitter",
    "split_into_sentences",
    "evaluate_chunk_quality",
    "compute_chunk_coherence",
    "ChunkQualityReport",
]
