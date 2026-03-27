"""Compatibility shim for legacy imports.

Canonical location:
`app.presentation.http.deps`
"""

from app.presentation.http.deps import get_ingestion_service, get_rag_service, get_session

__all__ = ["get_ingestion_service", "get_rag_service", "get_session"]
