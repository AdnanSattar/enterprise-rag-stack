"""
Semantic chunking module.
Chunking determines what the retriever can find.

Rules of thumb:
- Prefer semantic chunking over fixed length
- Aim for 150-300 words or 800-1500 tokens
- Use 5-15% overlap to preserve context
- Preserve hierarchical structure
"""

import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

import tiktoken

from .config import settings

logger = logging.getLogger(__name__)

# Initialize tokenizer
_enc = tiktoken.get_encoding("cl100k_base")


def num_tokens(text: str) -> int:
    """Count tokens using cl100k_base encoding."""
    return len(_enc.encode(text))


@dataclass
class Paragraph:
    """A paragraph with metadata."""

    text: str
    title: Optional[str] = None
    section: Optional[str] = None
    heading_level: int = 0


@dataclass
class Chunk:
    """A semantic chunk with metadata."""

    text: str
    titles: List[str]
    sections: List[str]
    token_count: int
    chunk_index: int


def extract_paragraphs(text: str) -> List[Paragraph]:
    """
    Extract paragraphs with heading detection.

    Preserves structure by identifying:
    - Markdown headings (# ## ###)
    - Section numbers (1.2.3)
    - All-caps headings
    """
    paragraphs = []
    current_title = None
    current_section = None

    # Split on double newlines
    blocks = re.split(r"\n\s*\n", text)

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        # Check for markdown heading
        heading_match = re.match(r"^(#{1,6})\s+(.+)$", block)
        if heading_match:
            level = len(heading_match.group(1))
            current_title = heading_match.group(2).strip()
            continue

        # Check for section number pattern
        section_match = re.match(r"^(\d+(?:\.\d+)*)\s+(.+)$", block)
        if section_match:
            current_section = section_match.group(1)
            block = section_match.group(2)

        # Check for all-caps heading (short line, all caps)
        if len(block) < 100 and block.isupper():
            current_title = block
            continue

        paragraphs.append(
            Paragraph(text=block, title=current_title, section=current_section)
        )

    return paragraphs


def semantic_chunk(
    paragraphs: List[Paragraph], max_tokens: int = None, overlap_tokens: int = None
) -> List[Chunk]:
    """
    Semantic chunking with hierarchical heading preservation.

    Args:
        paragraphs: List of paragraphs with metadata
        max_tokens: Maximum tokens per chunk (default from config)
        overlap_tokens: Overlap tokens between chunks (default from config)

    Returns:
        List of semantic chunks
    """
    max_tokens = max_tokens or settings.chunking.max_tokens
    overlap_tokens = overlap_tokens or settings.chunking.overlap_tokens

    chunks = []
    current: List[Paragraph] = []
    current_tokens = 0
    chunk_index = 0

    for para in paragraphs:
        p_tokens = num_tokens(para.text)

        # Handle oversized paragraphs
        if p_tokens > max_tokens:
            # Finalize current chunk if exists
            if current:
                chunks.append(_finalize_chunk(current, chunk_index))
                chunk_index += 1
                current = []
                current_tokens = 0

            # Split oversized paragraph by sentences
            sentences = re.split(r"(?<=[.!?])\s+", para.text)
            sub_current = []
            sub_tokens = 0

            for sent in sentences:
                s_tokens = num_tokens(sent)
                if sub_tokens + s_tokens > max_tokens and sub_current:
                    # Create chunk from sentences
                    combined = Paragraph(
                        text=" ".join(sub_current),
                        title=para.title,
                        section=para.section,
                    )
                    chunks.append(_finalize_chunk([combined], chunk_index))
                    chunk_index += 1
                    sub_current = []
                    sub_tokens = 0

                sub_current.append(sent)
                sub_tokens += s_tokens

            if sub_current:
                combined = Paragraph(
                    text=" ".join(sub_current), title=para.title, section=para.section
                )
                current = [combined]
                current_tokens = sub_tokens
            continue

        # Check if adding this paragraph exceeds limit
        if current and current_tokens + p_tokens > max_tokens:
            # Finalize current chunk
            chunks.append(_finalize_chunk(current, chunk_index))
            chunk_index += 1

            # Create overlap from end of current chunk
            overlap = []
            overlap_tok = 0
            for p in reversed(current):
                overlap.insert(0, p)
                overlap_tok += num_tokens(p.text)
                if overlap_tok >= overlap_tokens:
                    break

            current = overlap
            current_tokens = sum(num_tokens(p.text) for p in current)

        current.append(para)
        current_tokens += p_tokens

    # Don't forget the last chunk
    if current:
        chunks.append(_finalize_chunk(current, chunk_index))

    return chunks


def _finalize_chunk(paragraphs: List[Paragraph], index: int) -> Chunk:
    """Create a Chunk from a list of paragraphs."""
    text = "\n\n".join(p.text for p in paragraphs)
    titles = list({p.title for p in paragraphs if p.title})
    sections = list({p.section for p in paragraphs if p.section})

    return Chunk(
        text=text,
        titles=titles,
        sections=sections,
        token_count=num_tokens(text),
        chunk_index=index,
    )


def simple_chunk(
    text: str, max_tokens: int = None, overlap_tokens: int = None
) -> List[Dict]:
    """
    Simple chunking for unstructured text.
    Falls back to sentence-based splitting.

    Returns list of dicts for compatibility.
    """
    max_tokens = max_tokens or settings.chunking.max_tokens
    overlap_tokens = overlap_tokens or settings.chunking.overlap_tokens

    paragraphs = extract_paragraphs(text)
    chunks = semantic_chunk(paragraphs, max_tokens, overlap_tokens)

    return [
        {
            "text": c.text,
            "titles": c.titles,
            "sections": c.sections,
            "token_count": c.token_count,
            "chunk_index": c.chunk_index,
        }
        for c in chunks
    ]


def chunk_document(
    text: str, doc_id: str, metadata: Optional[Dict] = None
) -> List[Dict]:
    """
    Chunk a document and attach metadata to each chunk.

    Args:
        text: Document text
        doc_id: Document identifier
        metadata: Additional metadata to attach

    Returns:
        List of chunk dicts with IDs and metadata
    """
    base_metadata = metadata or {}
    chunks = simple_chunk(text)

    result = []
    for i, chunk in enumerate(chunks):
        chunk_id = f"{doc_id}#chunk_{i}"
        result.append(
            {
                "id": chunk_id,
                "doc_id": doc_id,
                "chunk_index": i,
                "text": chunk["text"],
                "titles": chunk["titles"],
                "sections": chunk["sections"],
                "token_count": chunk["token_count"],
                **base_metadata,
            }
        )

    logger.info(f"Document {doc_id} chunked into {len(result)} chunks")
    return result
