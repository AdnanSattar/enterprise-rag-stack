"""
Audit logging for RAG operations.

Logs all queries, retrievals, and document accesses for compliance.
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Optional


class AuditLogger:
    """
    Audit logger for RAG operations.

    Logs:
    - Query requests
    - Document retrievals
    - Document ingestions
    - Access violations
    """

    def __init__(self, log_file: Optional[str] = None):
        """
        Args:
            log_file: Optional file path for audit logs
        """
        self.logger = logging.getLogger("audit")
        self.log_file = log_file

        if log_file:
            handler = logging.FileHandler(log_file)
            handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def log_query(
        self,
        user_id: str,
        query: str,
        retrieved_doc_ids: List[str],
        tenant_id: Optional[str] = None,
    ):
        """Log a query request."""
        log_entry = {
            "event": "query",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "user_id": user_id,
            "tenant_id": tenant_id,
            "query": query,
            "retrieved_doc_ids": retrieved_doc_ids,
            "retrieved_count": len(retrieved_doc_ids),
        }
        self.logger.info(json.dumps(log_entry))

    def log_ingestion(self, user_id: str, doc_id: str, tenant_id: Optional[str] = None):
        """Log document ingestion."""
        log_entry = {
            "event": "ingestion",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "user_id": user_id,
            "tenant_id": tenant_id,
            "doc_id": doc_id,
        }
        self.logger.info(json.dumps(log_entry))

    def log_access_violation(
        self, user_id: str, attempted_doc_id: str, tenant_id: Optional[str] = None
    ):
        """Log access violation attempt."""
        log_entry = {
            "event": "access_violation",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "user_id": user_id,
            "tenant_id": tenant_id,
            "attempted_doc_id": attempted_doc_id,
        }
        self.logger.warning(json.dumps(log_entry))


# Global audit logger
_audit_logger: Optional[AuditLogger] = None


def get_audit_logger(log_file: Optional[str] = None) -> AuditLogger:
    """Get global audit logger."""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger(log_file=log_file)
    return _audit_logger
