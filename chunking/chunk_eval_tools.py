"""
Chunk quality evaluation tools.

Helps tune chunking parameters by measuring:
- Chunk coherence (semantic consistency)
- Size distribution
- Overlap effectiveness
- Retrieval recall impact
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ChunkQualityReport:
    """Report on chunk quality metrics."""

    total_chunks: int
    avg_tokens: float
    min_tokens: int
    max_tokens: int
    std_tokens: float
    avg_coherence: float
    chunks_too_small: int  # Below min threshold
    chunks_too_large: int  # Above max threshold
    overlap_quality: float
    recommendations: List[str] = field(default_factory=list)


def compute_chunk_coherence(
    chunk_text: str,
    embedding_service=None,
) -> float:
    """
    Compute semantic coherence within a chunk.

    Higher coherence means the chunk contains related content.
    Low coherence might indicate poor chunking boundaries.

    Args:
        chunk_text: Text of the chunk
        embedding_service: Optional embedding service

    Returns:
        Coherence score (0-1)
    """
    # Split into sentences
    import re

    sentences = re.split(r"(?<=[.!?])\s+", chunk_text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if len(sentences) < 2:
        return 1.0  # Single sentence is coherent by definition

    if embedding_service is None:
        # Fallback: simple lexical coherence
        return _lexical_coherence(sentences)

    try:
        # Semantic coherence via embeddings
        result = embedding_service.embed_batch(sentences)
        vectors = result.vectors

        # Compute pairwise similarities
        similarities = []
        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):
                sim = np.dot(vectors[i], vectors[j])
                similarities.append(sim)

        return float(np.mean(similarities)) if similarities else 1.0

    except Exception as e:
        logger.warning(f"Semantic coherence failed: {e}")
        return _lexical_coherence(sentences)


def _lexical_coherence(sentences: List[str]) -> float:
    """Simple lexical overlap coherence."""
    if len(sentences) < 2:
        return 1.0

    overlaps = []
    for i in range(len(sentences) - 1):
        words1 = set(sentences[i].lower().split())
        words2 = set(sentences[i + 1].lower().split())

        if words1 and words2:
            overlap = len(words1 & words2) / min(len(words1), len(words2))
            overlaps.append(overlap)

    return float(np.mean(overlaps)) if overlaps else 0.5


def evaluate_chunk_quality(
    chunks: List[Dict],
    min_tokens: int = 50,
    max_tokens: int = 1500,
    target_tokens: int = 800,
    embedding_service=None,
) -> ChunkQualityReport:
    """
    Evaluate overall chunking quality.

    Args:
        chunks: List of chunk dicts with 'text' and 'token_count'
        min_tokens: Minimum acceptable tokens
        max_tokens: Maximum acceptable tokens
        target_tokens: Target token count
        embedding_service: For semantic coherence

    Returns:
        ChunkQualityReport with metrics and recommendations
    """
    if not chunks:
        return ChunkQualityReport(
            total_chunks=0,
            avg_tokens=0,
            min_tokens=0,
            max_tokens=0,
            std_tokens=0,
            avg_coherence=0,
            chunks_too_small=0,
            chunks_too_large=0,
            overlap_quality=0,
            recommendations=["No chunks to evaluate"],
        )

    # Extract token counts
    token_counts = [
        c.get("token_count", len(c.get("text", "").split())) for c in chunks
    ]

    # Size statistics
    avg_tokens = float(np.mean(token_counts))
    min_tok = int(np.min(token_counts))
    max_tok = int(np.max(token_counts))
    std_tokens = float(np.std(token_counts))

    # Count problematic chunks
    too_small = sum(1 for t in token_counts if t < min_tokens)
    too_large = sum(1 for t in token_counts if t > max_tokens)

    # Compute coherence for sample
    sample_size = min(10, len(chunks))
    sample_indices = np.random.choice(len(chunks), sample_size, replace=False)
    coherence_scores = []

    for idx in sample_indices:
        text = chunks[idx].get("text", "")
        score = compute_chunk_coherence(text, embedding_service)
        coherence_scores.append(score)

    avg_coherence = float(np.mean(coherence_scores))

    # Evaluate overlap quality (if consecutive chunks available)
    overlap_quality = _evaluate_overlap(chunks)

    # Generate recommendations
    recommendations = []

    if too_small > len(chunks) * 0.1:
        recommendations.append(
            f"Consider reducing min_chunk_size. {too_small} chunks ({too_small/len(chunks)*100:.0f}%) "
            f"are below {min_tokens} tokens."
        )

    if too_large > len(chunks) * 0.1:
        recommendations.append(
            f"Consider reducing max_tokens. {too_large} chunks ({too_large/len(chunks)*100:.0f}%) "
            f"exceed {max_tokens} tokens."
        )

    if std_tokens > target_tokens * 0.5:
        recommendations.append(
            f"High variance in chunk sizes (std={std_tokens:.0f}). "
            "Consider more consistent chunking boundaries."
        )

    if avg_coherence < 0.5:
        recommendations.append(
            f"Low coherence score ({avg_coherence:.2f}). "
            "Chunks may span unrelated topics. Consider semantic chunking."
        )

    if overlap_quality < 0.3:
        recommendations.append(
            f"Low overlap quality ({overlap_quality:.2f}). "
            "Consider increasing overlap_tokens for better context preservation."
        )

    if not recommendations:
        recommendations.append("Chunking quality looks good!")

    return ChunkQualityReport(
        total_chunks=len(chunks),
        avg_tokens=avg_tokens,
        min_tokens=min_tok,
        max_tokens=max_tok,
        std_tokens=std_tokens,
        avg_coherence=avg_coherence,
        chunks_too_small=too_small,
        chunks_too_large=too_large,
        overlap_quality=overlap_quality,
        recommendations=recommendations,
    )


def _evaluate_overlap(chunks: List[Dict]) -> float:
    """
    Evaluate if overlap is effectively preserving context.

    Checks if consecutive chunks share meaningful content.
    """
    if len(chunks) < 2:
        return 1.0

    overlaps = []
    for i in range(len(chunks) - 1):
        text1 = chunks[i].get("text", "")
        text2 = chunks[i + 1].get("text", "")

        # Check word overlap in last/first portions
        words1 = text1.split()[-50:]  # Last 50 words
        words2 = text2.split()[:50]  # First 50 words

        set1 = set(w.lower() for w in words1)
        set2 = set(w.lower() for w in words2)

        if set1 and set2:
            overlap = len(set1 & set2) / min(len(set1), len(set2))
            overlaps.append(overlap)

    return float(np.mean(overlaps)) if overlaps else 0.0


def suggest_chunk_params(
    sample_text: str,
    domain: str = "general",
) -> Dict:
    """
    Suggest chunking parameters based on domain.

    Args:
        sample_text: Sample document text
        domain: Domain type (legal, code, faq, general)

    Returns:
        Suggested parameters
    """
    # Domain-specific defaults
    domain_params = {
        "legal": {
            "max_tokens": 1200,
            "overlap_tokens": 150,
            "reason": "Legal docs need larger chunks for complete clauses",
        },
        "code": {
            "max_tokens": 600,
            "overlap_tokens": 50,
            "reason": "Code should chunk on function/class boundaries",
        },
        "faq": {
            "max_tokens": 300,
            "overlap_tokens": 30,
            "reason": "FAQs have discrete Q&A pairs",
        },
        "general": {
            "max_tokens": 800,
            "overlap_tokens": 100,
            "reason": "Balanced for general content",
        },
    }

    base = domain_params.get(domain, domain_params["general"])

    # Analyze sample text
    import tiktoken

    enc = tiktoken.get_encoding("cl100k_base")

    total_tokens = len(enc.encode(sample_text))
    paragraphs = sample_text.split("\n\n")
    avg_para_tokens = total_tokens / max(len(paragraphs), 1)

    # Adjust based on content
    if avg_para_tokens > base["max_tokens"]:
        base["max_tokens"] = int(avg_para_tokens * 1.2)
        base["reason"] += ". Increased due to long paragraphs."

    return base
