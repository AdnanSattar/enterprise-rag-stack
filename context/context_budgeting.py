"""
Context token budget management.

Treat prompt context as a resource with a budget.

Considerations:
- Model context window limits
- Cost per token
- Response quality vs context length tradeoff
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import tiktoken

logger = logging.getLogger(__name__)

_enc = tiktoken.get_encoding("cl100k_base")


@dataclass
class ContextBudget:
    """Token budget allocation."""
    
    total_tokens: int
    system_prompt_tokens: int
    context_tokens: int
    query_tokens: int
    response_tokens: int
    
    @property
    def available_for_context(self) -> int:
        """Tokens available for retrieved context."""
        return (
            self.total_tokens -
            self.system_prompt_tokens -
            self.query_tokens -
            self.response_tokens
        )


def allocate_token_budget(
    model_max_tokens: int = 128000,
    system_prompt: str = None,
    query: str = None,
    expected_response_tokens: int = 1024,
    context_ratio: float = 0.6,
) -> ContextBudget:
    """
    Allocate token budget for a prompt.
    
    Args:
        model_max_tokens: Model's context window
        system_prompt: System prompt text
        query: User query
        expected_response_tokens: Tokens reserved for response
        context_ratio: Ratio of remaining tokens for context
        
    Returns:
        ContextBudget allocation
    """
    system_tokens = len(_enc.encode(system_prompt)) if system_prompt else 0
    query_tokens = len(_enc.encode(query)) if query else 0
    
    remaining = model_max_tokens - system_tokens - query_tokens - expected_response_tokens
    context_tokens = int(remaining * context_ratio)
    
    return ContextBudget(
        total_tokens=model_max_tokens,
        system_prompt_tokens=system_tokens,
        context_tokens=context_tokens,
        query_tokens=query_tokens,
        response_tokens=expected_response_tokens,
    )


def truncate_to_budget(
    text: str,
    max_tokens: int,
    strategy: str = "end",
) -> str:
    """
    Truncate text to fit token budget.
    
    Strategies:
    - end: Truncate from end
    - middle: Keep start and end, truncate middle
    - smart: Try to truncate at sentence boundaries
    
    Args:
        text: Text to truncate
        max_tokens: Maximum tokens
        strategy: Truncation strategy
        
    Returns:
        Truncated text
    """
    tokens = _enc.encode(text)
    
    if len(tokens) <= max_tokens:
        return text
    
    if strategy == "end":
        return _enc.decode(tokens[:max_tokens])
    
    elif strategy == "middle":
        # Keep first 40%, last 40%, skip middle
        start_tokens = int(max_tokens * 0.4)
        end_tokens = max_tokens - start_tokens
        
        start = _enc.decode(tokens[:start_tokens])
        end = _enc.decode(tokens[-end_tokens:])
        
        return f"{start}\n\n[...content truncated...]\n\n{end}"
    
    elif strategy == "smart":
        # Try to truncate at sentence boundary
        text_truncated = _enc.decode(tokens[:max_tokens])
        
        # Find last sentence boundary
        for marker in [". ", ".\n", "? ", "! "]:
            idx = text_truncated.rfind(marker)
            if idx > len(text_truncated) * 0.5:  # Don't truncate too much
                return text_truncated[:idx + 1]
        
        return text_truncated
    
    return _enc.decode(tokens[:max_tokens])


def estimate_cost(
    prompt_tokens: int,
    completion_tokens: int,
    model: str = "gpt-4",
) -> Dict[str, float]:
    """
    Estimate API cost for a request.
    
    Prices as of 2025 (update as needed):
    - GPT-4: $0.03/1K input, $0.06/1K output
    - GPT-4 Turbo: $0.01/1K input, $0.03/1K output
    - GPT-3.5: $0.0015/1K input, $0.002/1K output
    
    Args:
        prompt_tokens: Input tokens
        completion_tokens: Output tokens
        model: Model name
        
    Returns:
        Cost breakdown
    """
    pricing = {
        "gpt-4": (0.03, 0.06),
        "gpt-4-turbo": (0.01, 0.03),
        "gpt-4o": (0.005, 0.015),
        "gpt-4o-mini": (0.00015, 0.0006),
        "gpt-3.5-turbo": (0.0015, 0.002),
    }
    
    # Find matching pricing
    input_price, output_price = pricing.get(model, (0.01, 0.03))
    
    for key in pricing:
        if key in model.lower():
            input_price, output_price = pricing[key]
            break
    
    input_cost = (prompt_tokens / 1000) * input_price
    output_cost = (completion_tokens / 1000) * output_price
    
    return {
        "model": model,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "input_cost_usd": input_cost,
        "output_cost_usd": output_cost,
        "total_cost_usd": input_cost + output_cost,
    }


def select_chunk_count(
    chunks: List,
    max_tokens: int,
    min_chunks: int = 3,
    max_chunks: int = 10,
) -> int:
    """
    Select optimal number of chunks for context.
    
    Balance: More chunks = more info but less depth per chunk.
    
    Args:
        chunks: Available chunks
        max_tokens: Token budget
        min_chunks: Minimum chunks to include
        max_chunks: Maximum chunks to include
        
    Returns:
        Optimal number of chunks
    """
    if not chunks:
        return 0
    
    # Estimate tokens per chunk
    avg_tokens = sum(
        len(_enc.encode(c.text if hasattr(c, 'text') else c.get('text', '')))
        for c in chunks[:5]
    ) / min(5, len(chunks))
    
    # Estimate with header overhead
    tokens_per_chunk = avg_tokens + 50  # Header overhead
    
    # Calculate optimal count
    optimal = int(max_tokens / tokens_per_chunk)
    
    return max(min_chunks, min(optimal, max_chunks, len(chunks)))

