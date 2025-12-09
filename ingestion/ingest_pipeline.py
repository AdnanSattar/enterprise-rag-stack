"""
Production ingestion pipeline.

Orchestrates the full document ingestion workflow:
Raw files -> Parsers (PDF, HTML, Office) -> Cleaning & dedupe -> Normalized text + metadata -> Chunking
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from .dedupe import ContentDeduplicator
from .normalize import NormalizationResult, normalize_document

logger = logging.getLogger(__name__)


@dataclass
class IngestionStats:
    """Statistics from an ingestion run."""

    total_docs: int = 0
    successful: int = 0
    failed: int = 0
    duplicates_removed: int = 0
    total_chars_processed: int = 0
    total_words_processed: int = 0


@dataclass
class IngestionConfig:
    """Configuration for ingestion pipeline."""

    # File handling
    supported_extensions: List[str] = field(
        default_factory=lambda: [".txt", ".md", ".pdf", ".docx", ".html"]
    )
    max_file_size_mb: float = 50.0

    # Processing
    enable_ocr: bool = True
    enable_semantic_dedup: bool = False
    semantic_dedup_threshold: float = 0.95

    # Custom patterns
    custom_boilerplate_patterns: List[str] = field(default_factory=list)


class IngestionPipeline:
    """
    Production ingestion pipeline with deduplication.

    Flow:
    Raw files -> Parsers -> Cleaning & dedupe -> Normalized text + metadata -> Ready for chunking

    Usage:
        pipeline = IngestionPipeline(tenant_id="acme")

        # Process single file
        result = pipeline.process_file("./contract.pdf")

        # Process directory
        results = pipeline.process_directory("./documents")

        # Get statistics
        stats = pipeline.get_stats()
    """

    def __init__(
        self,
        tenant_id: str,
        config: Optional[IngestionConfig] = None,
    ):
        """
        Args:
            tenant_id: Tenant identifier for multi-tenancy
            config: Pipeline configuration
        """
        self.tenant_id = tenant_id
        self.config = config or IngestionConfig()
        self.deduplicator = ContentDeduplicator(
            enable_semantic_dedup=self.config.enable_semantic_dedup,
            similarity_threshold=self.config.semantic_dedup_threshold,
        )
        self.stats = IngestionStats()

    def process_file(
        self,
        path: str,
        metadata: Optional[Dict] = None,
    ) -> Optional[NormalizationResult]:
        """
        Process a single file with deduplication.

        Args:
            path: Path to file
            metadata: Additional metadata

        Returns:
            NormalizationResult if successful and not duplicate, None otherwise
        """
        self.stats.total_docs += 1
        file_path = Path(path)

        # Check file size
        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            if size_mb > self.config.max_file_size_mb:
                logger.warning(f"File too large ({size_mb:.1f}MB): {path}")
                self.stats.failed += 1
                return None

        # Normalize document
        result = normalize_document(
            path=path,
            tenant_id=self.tenant_id,
            additional_metadata=metadata,
            custom_boilerplate=self.config.custom_boilerplate_patterns,
        )

        if not result.success:
            self.stats.failed += 1
            logger.error(f"Failed to process {path}: {result.errors}")
            return None

        # Check for duplicates
        dedup_result = self.deduplicator.check(result.doc_id, result.text)
        if dedup_result.is_duplicate:
            self.stats.duplicates_removed += 1
            logger.info(
                f"Duplicate detected, skipping: {path} "
                f"(matches {dedup_result.existing_doc_id})"
            )
            return None

        # Add to dedup index
        self.deduplicator.add(result.doc_id, result.doc_id, result.text)

        # Update stats
        self.stats.successful += 1
        self.stats.total_chars_processed += len(result.text)
        self.stats.total_words_processed += len(result.text.split())

        return result

    def process_directory(
        self,
        directory: str,
        extensions: Optional[List[str]] = None,
        recursive: bool = True,
        max_files: Optional[int] = None,
    ) -> List[NormalizationResult]:
        """
        Process all documents in a directory.

        Args:
            directory: Directory path
            extensions: File extensions to process (uses config default if None)
            recursive: Search subdirectories
            max_files: Maximum files to process (None for unlimited)

        Returns:
            List of successful NormalizationResults
        """
        extensions = extensions or self.config.supported_extensions
        dir_path = Path(directory)

        if not dir_path.exists():
            logger.error(f"Directory not found: {directory}")
            return []

        pattern = "**/*" if recursive else "*"
        results = []
        processed = 0

        for file_path in dir_path.glob(pattern):
            if max_files and processed >= max_files:
                logger.info(f"Reached max_files limit ({max_files})")
                break

            if file_path.is_file() and file_path.suffix.lower() in extensions:
                result = self.process_file(str(file_path))
                if result:
                    results.append(result)
                processed += 1

        logger.info(
            f"Ingestion complete: {self.stats.successful}/{self.stats.total_docs} "
            f"successful, {self.stats.duplicates_removed} duplicates removed"
        )

        return results

    def process_batch(
        self,
        file_paths: List[str],
        metadata: Optional[Dict] = None,
    ) -> List[NormalizationResult]:
        """
        Process a batch of files.

        Args:
            file_paths: List of file paths
            metadata: Metadata to apply to all files

        Returns:
            List of successful results
        """
        results = []
        for path in file_paths:
            result = self.process_file(path, metadata)
            if result:
                results.append(result)
        return results

    def get_stats(self) -> Dict:
        """Get ingestion statistics."""
        return {
            "total_docs": self.stats.total_docs,
            "successful": self.stats.successful,
            "failed": self.stats.failed,
            "duplicates_removed": self.stats.duplicates_removed,
            "total_chars_processed": self.stats.total_chars_processed,
            "total_words_processed": self.stats.total_words_processed,
            "dedup_stats": self.deduplicator.get_stats(),
        }

    def reset_stats(self) -> None:
        """Reset statistics for a new run."""
        self.stats = IngestionStats()

    def reset_dedup_index(self) -> None:
        """Clear the deduplication index."""
        self.deduplicator.clear()


# Convenience function for CLI usage
def run_ingestion(
    input_path: str,
    tenant_id: str,
    recursive: bool = True,
    extensions: Optional[List[str]] = None,
) -> Dict:
    """
    Run ingestion from command line.

    Args:
        input_path: File or directory path
        tenant_id: Tenant identifier
        recursive: Search subdirectories
        extensions: File extensions

    Returns:
        Ingestion statistics
    """
    pipeline = IngestionPipeline(tenant_id=tenant_id)

    path = Path(input_path)
    if path.is_file():
        result = pipeline.process_file(str(path))
        return {
            "mode": "single_file",
            "success": result is not None,
            "stats": pipeline.get_stats(),
        }
    elif path.is_dir():
        results = pipeline.process_directory(
            str(path),
            extensions=extensions,
            recursive=recursive,
        )
        return {
            "mode": "directory",
            "documents_processed": len(results),
            "stats": pipeline.get_stats(),
        }
    else:
        return {"error": f"Path not found: {input_path}"}


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Run document ingestion")
    parser.add_argument("--input", required=True, help="Input file or directory")
    parser.add_argument("--tenant", default="default", help="Tenant ID")
    parser.add_argument("--recursive", action="store_true", help="Search subdirs")

    args = parser.parse_args()

    result = run_ingestion(
        input_path=args.input,
        tenant_id=args.tenant,
        recursive=args.recursive,
    )

    print(json.dumps(result, indent=2))
