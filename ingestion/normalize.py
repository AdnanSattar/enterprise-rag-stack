"""
Document normalization and cleaning.

Clean input beats clever retrieval. This module handles:
- Boilerplate removal
- OCR artifact cleaning
- Metadata extraction
- Text normalization

Pipeline: Raw files -> Parsers -> Cleaning -> Normalized text + metadata
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Common boilerplate patterns to remove
BOILERPLATE_PATTERNS = [
    r"(?i)confidential\s+information",
    r"(?i)all\s+rights\s+reserved",
    r"(?i)page\s+\d+\s+of\s+\d+",
    r"(?i)copyright\s+©?\s*\d{4}",
    r"(?i)proprietary\s+and\s+confidential",
    r"(?i)do\s+not\s+distribute",
    r"(?i)internal\s+use\s+only",
    r"(?i)draft\s*-?\s*not\s+for\s+distribution",
]


@dataclass
class NormalizationResult:
    """Result of document normalization."""

    doc_id: str
    text: str
    metadata: Dict
    success: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def strip_boilerplate(text: str, patterns: List[str] = None) -> str:
    """
    Remove boilerplate text that pollutes retrieval.

    Args:
        text: Raw text to clean
        patterns: Regex patterns to remove (uses defaults if None)

    Returns:
        Cleaned text with boilerplate removed

    Example:
        >>> text = "Page 1 of 10\\nActual content here\\nAll rights reserved"
        >>> clean = strip_boilerplate(text)
        >>> "Page 1 of 10" not in clean
        True
    """
    patterns = patterns or BOILERPLATE_PATTERNS

    for pat in patterns:
        text = re.sub(pat, " ", text)

    # Normalize excessive whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)

    return text.strip()


def clean_ocr_artifacts(text: str) -> str:
    """
    Clean common OCR artifacts.

    Handles:
    - Broken words across lines (word-\\nbreak -> wordbreak)
    - Random line breaks mid-sentence
    - Unicode quote/dash normalization
    - Page break artifacts

    Args:
        text: Text with potential OCR artifacts

    Returns:
        Cleaned text
    """
    # Fix hyphenated line breaks (word-\nbreak -> wordbreak)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    # Fix orphaned single characters at line ends
    text = re.sub(r"\s([a-zA-Z])\n([a-zA-Z])", r" \1\2", text)

    # Normalize unicode quotes and dashes
    text = text.replace(""", '"').replace(""", '"')
    text = text.replace("'", "'").replace("'", "'")
    text = text.replace("–", "-").replace("—", "-")

    # Remove page break artifacts
    text = re.sub(r"\f", "\n\n", text)

    return text


def extract_metadata_from_path(path: Path) -> Dict:
    """
    Extract metadata from file path and stats.

    Args:
        path: Path object for the file

    Returns:
        Dict containing file metadata
    """
    stat = path.stat()

    return {
        "source_path": str(path.absolute()),
        "filename": path.name,
        "file_extension": path.suffix.lower(),
        "filesize_bytes": stat.st_size,
        "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat() + "Z",
        "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat() + "Z",
    }


def normalize_document(
    path: str,
    tenant_id: str,
    additional_metadata: Optional[Dict] = None,
    custom_boilerplate: Optional[List[str]] = None,
) -> NormalizationResult:
    """
    Full document normalization pipeline.

    Pipeline:
    1. Read file content (using unstructured if available)
    2. Clean OCR artifacts
    3. Remove boilerplate
    4. Extract metadata
    5. Generate content hash for dedup

    Args:
        path: Path to document file
        tenant_id: Tenant identifier for multi-tenancy
        additional_metadata: Extra metadata to include
        custom_boilerplate: Additional boilerplate patterns to remove

    Returns:
        NormalizationResult with cleaned text and metadata

    Example:
        >>> result = normalize_document("./contract.pdf", "acme")
        >>> if result.success:
        ...     print(f"Processed {result.metadata['filename']}")
    """
    from .dedupe import compute_content_hash

    errors = []
    warnings = []
    file_path = Path(path)

    # Check file exists
    if not file_path.exists():
        return NormalizationResult(
            doc_id="",
            text="",
            metadata={},
            success=False,
            errors=[f"File not found: {path}"],
        )

    try:
        # Try to use unstructured if available (handles PDFs, DOCX, etc.)
        try:
            from unstructured.partition.auto import partition

            elements = partition(filename=str(file_path))
            text_blocks = [e.text for e in elements if getattr(e, "text", None)]
            raw_text = "\n\n".join(text_blocks)
        except ImportError:
            # Fallback to simple text read
            warnings.append("unstructured not installed, using basic text read")
            raw_text = file_path.read_text(encoding="utf-8", errors="replace")

        # Clean the text
        cleaned = clean_ocr_artifacts(raw_text)

        # Remove boilerplate
        patterns = BOILERPLATE_PATTERNS.copy()
        if custom_boilerplate:
            patterns.extend(custom_boilerplate)
        cleaned = strip_boilerplate(cleaned, patterns)

        # Generate content hash as doc_id
        doc_id = compute_content_hash(cleaned)

        # Build metadata
        metadata = extract_metadata_from_path(file_path)
        metadata.update(
            {
                "doc_id": doc_id,
                "tenant_id": tenant_id,
                "ingested_at": datetime.utcnow().isoformat() + "Z",
                "content_hash": doc_id,
                "char_count": len(cleaned),
                "word_count": len(cleaned.split()),
            }
        )

        if additional_metadata:
            metadata.update(additional_metadata)

        return NormalizationResult(
            doc_id=doc_id,
            text=cleaned,
            metadata=metadata,
            success=True,
            errors=errors,
            warnings=warnings,
        )

    except Exception as e:
        logger.exception(f"Error processing {path}")
        return NormalizationResult(
            doc_id="", text="", metadata={}, success=False, errors=[str(e)]
        )
