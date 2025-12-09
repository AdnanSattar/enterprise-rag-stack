"""
Document Ingestion Module.

This module handles the complete document ingestion pipeline:
- Raw file parsing (PDF, HTML, Office, etc.)
- Text cleaning and normalization
- Boilerplate removal
- Content deduplication
- Metadata extraction

Usage:
    from ingestion import IngestionPipeline, normalize_document

    pipeline = IngestionPipeline(tenant_id="acme")
    results = pipeline.process_directory("./documents")
"""

from .dedupe import ContentDeduplicator, compute_content_hash
from .ingest_pipeline import IngestionPipeline, IngestionStats
from .normalize import (
    clean_ocr_artifacts,
    extract_metadata_from_path,
    normalize_document,
    strip_boilerplate,
)

__all__ = [
    "IngestionPipeline",
    "IngestionStats",
    "normalize_document",
    "strip_boilerplate",
    "clean_ocr_artifacts",
    "extract_metadata_from_path",
    "compute_content_hash",
    "ContentDeduplicator",
]
