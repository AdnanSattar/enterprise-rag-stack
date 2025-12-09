"""
Sentence-level text splitting utilities.

For fine-grained chunking when semantic paragraph
boundaries aren't available or appropriate.
"""

import logging
import re
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Sentence:
    """A sentence with position information."""

    text: str
    start: int
    end: int
    index: int


def split_into_sentences(text: str) -> List[Sentence]:
    """
    Split text into sentences.

    Handles common edge cases:
    - Abbreviations (Mr., Dr., etc.)
    - Numbers with decimals
    - Ellipsis
    - Question marks and exclamation points

    Args:
        text: Text to split

    Returns:
        List of Sentence objects
    """
    # Common abbreviations that shouldn't split
    abbreviations = {
        "Mr",
        "Mrs",
        "Ms",
        "Dr",
        "Prof",
        "Sr",
        "Jr",
        "vs",
        "etc",
        "eg",
        "ie",
        "al",
        "Inc",
        "Ltd",
        "Corp",
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    }

    # Protect abbreviations
    protected_text = text
    for abbr in abbreviations:
        protected_text = re.sub(
            rf"\b({abbr})\.", rf"\1<DOT>", protected_text, flags=re.IGNORECASE
        )

    # Protect decimal numbers
    protected_text = re.sub(r"(\d)\.(\d)", r"\1<DOT>\2", protected_text)

    # Split on sentence boundaries
    pattern = r"(?<=[.!?])\s+"
    parts = re.split(pattern, protected_text)

    sentences = []
    current_pos = 0

    for i, part in enumerate(parts):
        # Restore protected dots
        restored = part.replace("<DOT>", ".")

        # Find actual position in original text
        start = text.find(restored, current_pos)
        if start == -1:
            start = current_pos
        end = start + len(restored)

        if restored.strip():
            sentences.append(
                Sentence(
                    text=restored.strip(),
                    start=start,
                    end=end,
                    index=i,
                )
            )

        current_pos = end

    return sentences


class SentenceSplitter:
    """
    Sentence-based text splitter with token-aware grouping.

    Usage:
        splitter = SentenceSplitter(max_tokens=500)
        chunks = splitter.split(long_text)
    """

    def __init__(
        self,
        max_tokens: int = 500,
        overlap_sentences: int = 1,
        min_sentences: int = 1,
    ):
        """
        Args:
            max_tokens: Maximum tokens per chunk
            overlap_sentences: Sentences to overlap between chunks
            min_sentences: Minimum sentences per chunk
        """
        self.max_tokens = max_tokens
        self.overlap_sentences = overlap_sentences
        self.min_sentences = min_sentences

        # Import tokenizer
        import tiktoken

        self._enc = tiktoken.get_encoding("cl100k_base")

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self._enc.encode(text))

    def split(self, text: str) -> List[str]:
        """
        Split text into token-limited chunks.

        Args:
            text: Text to split

        Returns:
            List of chunk strings
        """
        sentences = split_into_sentences(text)

        if not sentences:
            return [text] if text.strip() else []

        chunks = []
        current_sentences = []
        current_tokens = 0

        for sentence in sentences:
            sent_tokens = self._count_tokens(sentence.text)

            # Check if adding this sentence exceeds limit
            if current_tokens + sent_tokens > self.max_tokens and current_sentences:
                # Create chunk
                chunk_text = " ".join(s.text for s in current_sentences)
                chunks.append(chunk_text)

                # Keep overlap sentences
                if self.overlap_sentences > 0:
                    current_sentences = current_sentences[-self.overlap_sentences :]
                    current_tokens = sum(
                        self._count_tokens(s.text) for s in current_sentences
                    )
                else:
                    current_sentences = []
                    current_tokens = 0

            current_sentences.append(sentence)
            current_tokens += sent_tokens

        # Don't forget last chunk
        if current_sentences:
            chunk_text = " ".join(s.text for s in current_sentences)
            chunks.append(chunk_text)

        return chunks

    def split_with_metadata(self, text: str, doc_id: str = None) -> List[dict]:
        """
        Split text and return chunks with metadata.

        Args:
            text: Text to split
            doc_id: Optional document ID

        Returns:
            List of chunk dicts
        """
        chunks = self.split(text)

        return [
            {
                "text": chunk,
                "chunk_index": i,
                "doc_id": doc_id,
                "token_count": self._count_tokens(chunk),
            }
            for i, chunk in enumerate(chunks)
        ]


def split_by_tokens(
    text: str,
    max_tokens: int = 500,
    overlap_tokens: int = 50,
) -> List[str]:
    """
    Split text by token count (not semantic boundaries).

    Use only when semantic chunking isn't appropriate.

    Args:
        text: Text to split
        max_tokens: Maximum tokens per chunk
        overlap_tokens: Token overlap between chunks

    Returns:
        List of chunk strings
    """
    import tiktoken

    enc = tiktoken.get_encoding("cl100k_base")

    tokens = enc.encode(text)
    chunks = []

    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk_tokens = tokens[start:end]
        chunks.append(enc.decode(chunk_tokens))

        # Move start with overlap
        start = end - overlap_tokens if end < len(tokens) else end

    return chunks
