"""
ACL (Access Control List) filtering for multi-tenant RAG.

Ensures users can only retrieve documents they have access to.
"""

from typing import Dict, List, Optional, Set


class ACLFilter:
    """
    Filter documents based on access control lists.

    Supports:
    - Tenant-based isolation
    - User-based permissions
    - Document-level ACLs
    """

    def __init__(self):
        """Initialize ACL filter."""
        self._tenant_docs: Dict[str, Set[str]] = {}
        self._user_permissions: Dict[str, Set[str]] = {}

    def add_tenant_document(self, tenant_id: str, doc_id: str):
        """Register a document for a tenant."""
        if tenant_id not in self._tenant_docs:
            self._tenant_docs[tenant_id] = set()
        self._tenant_docs[tenant_id].add(doc_id)

    def grant_user_access(self, user_id: str, doc_id: str):
        """Grant a user access to a document."""
        if user_id not in self._user_permissions:
            self._user_permissions[user_id] = set()
        self._user_permissions[user_id].add(doc_id)

    def filter_by_tenant(self, doc_ids: List[str], tenant_id: str) -> List[str]:
        """
        Filter documents by tenant.

        Args:
            doc_ids: List of document IDs
            tenant_id: Tenant ID

        Returns:
            Filtered list of document IDs
        """
        if tenant_id not in self._tenant_docs:
            return []

        allowed_docs = self._tenant_docs[tenant_id]
        return [doc_id for doc_id in doc_ids if doc_id in allowed_docs]

    def filter_by_user(self, doc_ids: List[str], user_id: str) -> List[str]:
        """
        Filter documents by user permissions.

        Args:
            doc_ids: List of document IDs
            user_id: User ID

        Returns:
            Filtered list of document IDs
        """
        if user_id not in self._user_permissions:
            return []

        allowed_docs = self._user_permissions[user_id]
        return [doc_id for doc_id in doc_ids if doc_id in allowed_docs]

    def build_metadata_filter(
        self, tenant_id: Optional[str] = None, user_id: Optional[str] = None
    ) -> Dict:
        """
        Build metadata filter for vector store query.

        Args:
            tenant_id: Tenant ID
            user_id: User ID

        Returns:
            Metadata filter dict
        """
        filters = {}

        if tenant_id:
            filters["tenant_id"] = tenant_id

        if user_id:
            # In production, you'd query allowed doc_ids from ACL
            # For now, we rely on metadata filtering
            pass

        return filters


# Global ACL filter
_acl_filter: Optional[ACLFilter] = None


def get_acl_filter() -> ACLFilter:
    """Get global ACL filter."""
    global _acl_filter
    if _acl_filter is None:
        _acl_filter = ACLFilter()
    return _acl_filter
