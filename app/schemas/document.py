from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.db.models import DocumentStatus


class DocumentUploadResponse(BaseModel):
    document_id: UUID
    filename: str
    file_size_bytes: int
    status: DocumentStatus
    task_id: str | None = None
    message: str


class DocumentStatusResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    filename: str
    file_size_bytes: int
    status: DocumentStatus
    error_message: str | None = None
    page_count: int = 0
    chunk_count: int = 0
    created_at: datetime
    updated_at: datetime


class DocumentListResponse(BaseModel):
    items: list[DocumentStatusResponse] = Field(default_factory=list)


class DocumentDeleteRequest(BaseModel):
    document_ids: list[UUID] = Field(min_length=1)


class DocumentDeleteResponse(BaseModel):
    deleted_ids: list[UUID]
    deleted_count: int
