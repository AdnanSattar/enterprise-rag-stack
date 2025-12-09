"""
Context Assembly Module.

Treat prompt context as a resource with a budget.

This module handles:
- Context building from retrieved chunks
- Token budget management
- Prompt template formatting
- Inline citation formatting

Best practices:
- Concatenate top N reranked chunks with section headers
- Provide source citations inline using chunk IDs
- Truncate by token budget, prefer chunk density

Usage:
    from context import assemble_context, build_prompt

    context = assemble_context(chunks, max_tokens=3000)
    prompt = build_prompt(context.text, query, prompt_type="grounded_qa")
"""

from .context_budgeting import ContextBudget, allocate_token_budget, truncate_to_budget
from .context_builder import (
    AssembledContext,
    assemble_context,
    format_chunk_header,
    format_context_simple,
)

__all__ = [
    "assemble_context",
    "format_chunk_header",
    "format_context_simple",
    "AssembledContext",
    "ContextBudget",
    "allocate_token_budget",
    "truncate_to_budget",
]
