"""
Context assembly and prompt budget management.
Treat prompt context as a resource with a budget.

Best practices:
- Concatenate top N reranked chunks with section headers
- Provide source citations inline using chunk IDs
- Truncate by token budget, prefer chunk density
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import tiktoken

from .config import settings
from .reranking import RerankResult

# Initialize tokenizer
_enc = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Count tokens in text."""
    return len(_enc.encode(text))


@dataclass
class AssembledContext:
    """Assembled context ready for LLM prompt."""

    text: str
    sources: List[Dict]
    token_count: int
    chunks_included: int
    chunks_truncated: int


def format_chunk_header(
    chunk: RerankResult, index: int, include_score: bool = True
) -> str:
    """
    Format a chunk header for LLM context.

    Example:
    [Source 1] doc=contract_001 chunk=chunk_0 | score: 0.92
    Title: Payment Terms
    """
    parts = [f"[Source {index + 1}]"]
    parts.append(f"doc={chunk.doc_id}")
    parts.append(f"chunk={chunk.id}")

    if include_score:
        parts.append(f"| score: {chunk.final_score:.2f}")

    header = " ".join(parts)

    # Add title if available
    title = chunk.metadata.get("title") or (
        chunk.metadata.get("titles", [None])[0]
        if chunk.metadata.get("titles")
        else None
    )
    if title:
        header += f"\nTitle: {title}"

    return header


def assemble_context(
    chunks: List[RerankResult],
    max_tokens: int = None,
    include_scores: bool = True,
    chunk_separator: str = "\n\n---\n\n",
) -> AssembledContext:
    """
    Assemble context from reranked chunks.

    Args:
        chunks: Reranked and deduplicated chunks
        max_tokens: Maximum tokens for context (default: 3/4 of LLM max)
        include_scores: Include relevance scores in headers
        chunk_separator: Separator between chunks

    Returns:
        AssembledContext ready for prompt
    """
    max_tokens = max_tokens or int(settings.llm.max_tokens * 0.75)

    context_parts = []
    sources = []
    current_tokens = 0
    chunks_included = 0
    chunks_truncated = 0

    for i, chunk in enumerate(chunks):
        # Format this chunk
        header = format_chunk_header(chunk, i, include_scores)
        chunk_text = f"{header}\n\n{chunk.text}"
        chunk_tokens = count_tokens(chunk_text)

        # Check if we can fit this chunk
        separator_tokens = count_tokens(chunk_separator) if context_parts else 0
        total_new_tokens = chunk_tokens + separator_tokens

        if current_tokens + total_new_tokens > max_tokens:
            # Try to fit a truncated version
            remaining_tokens = max_tokens - current_tokens - separator_tokens
            if remaining_tokens > 200:  # Minimum useful chunk
                # Truncate the chunk text
                truncated = _truncate_to_tokens(
                    chunk.text, remaining_tokens - count_tokens(header) - 20
                )
                chunk_text = f"{header}\n\n{truncated}..."
                chunk_tokens = count_tokens(chunk_text)
                chunks_truncated += 1
            else:
                # Can't fit even truncated, stop adding
                break

        if context_parts:
            context_parts.append(chunk_separator)
            current_tokens += separator_tokens

        context_parts.append(chunk_text)
        current_tokens += chunk_tokens
        chunks_included += 1

        sources.append(
            {
                "doc_id": chunk.doc_id,
                "chunk_id": chunk.id,
                "score": chunk.final_score,
                "title": chunk.metadata.get("title"),
            }
        )

    return AssembledContext(
        text="".join(context_parts),
        sources=sources,
        token_count=current_tokens,
        chunks_included=chunks_included,
        chunks_truncated=chunks_truncated,
    )


def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate text to approximately max_tokens."""
    tokens = _enc.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return _enc.decode(tokens[:max_tokens])


def format_context_simple(chunks: List[Dict]) -> str:
    """
    Simple context formatting for basic use cases.

    Args:
        chunks: List of chunk dicts with 'doc_id', 'chunk_id', 'text'

    Returns:
        Formatted context string
    """
    out = []
    for i, ch in enumerate(chunks):
        header = f"[Chunk {i+1}] doc={ch.get('doc_id', 'unknown')} chunk={ch.get('chunk_id', ch.get('id', i))}"
        body = ch.get("text", "")
        out.append(header + "\n" + body)
    return "\n\n".join(out)


# Prompt templates
GROUNDED_QA_PROMPT = """You are an enterprise assistant. Use ONLY the context below to answer the question.

Context:
{context}

Question: {query}

Instructions:
- Answer based solely on the provided context
- If the answer is not explicitly in the context, reply: "Not found in context"
- Be concise and accurate
- Cite source numbers when making claims

Answer:"""


STRUCTURED_OUTPUT_PROMPT = """You are a compliance assistant. Use only the context below.

Context:
{context}

Question: {query}

If the requested fact is not present in the context, reply with answer "Not found in context".

Return EXACT JSON:
{{
  "answer": "<your answer>",
  "sources": [{{"doc_id": "<id>", "chunk_id": "<id>"}}],
  "confidence": "<low|medium|high>"
}}"""


SUMMARIZATION_PROMPT = """You are a document summarizer. Summarize the key points from the context.

Context:
{context}

Task: {query}

Provide a structured summary with:
1. Main points (bullet list)
2. Key facts and figures
3. Important caveats or limitations

Summary:"""


def build_prompt(context: str, query: str, prompt_type: str = "grounded_qa") -> str:
    """
    Build a prompt from context and query.

    Args:
        context: Assembled context string
        query: User query
        prompt_type: Type of prompt (grounded_qa, structured, summarization)

    Returns:
        Complete prompt string
    """
    templates = {
        "grounded_qa": GROUNDED_QA_PROMPT,
        "structured": STRUCTURED_OUTPUT_PROMPT,
        "summarization": SUMMARIZATION_PROMPT,
    }

    template = templates.get(prompt_type, GROUNDED_QA_PROMPT)
    return template.format(context=context, query=query)
