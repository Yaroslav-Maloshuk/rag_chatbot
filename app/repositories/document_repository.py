from __future__ import annotations

from collections.abc import Sequence
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Document, DocumentStatus


class DocumentRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(self, filename: str, file_path: str, file_size_bytes: int) -> Document:
        document = Document(
            filename=filename,
            file_path=file_path,
            file_size_bytes=file_size_bytes,
            status=DocumentStatus.UPLOADED,
        )
        self.session.add(document)
        await self.session.commit()
        await self.session.refresh(document)
        return document

    async def get(self, document_id: UUID) -> Document | None:
        return await self.session.get(Document, document_id)

    async def list(self, limit: int = 100) -> Sequence[Document]:
        stmt = select(Document).order_by(Document.created_at.desc()).limit(limit)
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def get_by_ids(self, document_ids: list[UUID]) -> Sequence[Document]:
        if not document_ids:
            return []
        stmt = select(Document).where(Document.id.in_(document_ids))
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def update_status(
        self,
        document: Document,
        status: DocumentStatus,
        *,
        error_message: str | None = None,
        page_count: int | None = None,
        chunk_count: int | None = None,
    ) -> Document:
        document.status = status
        document.error_message = error_message
        if page_count is not None:
            document.page_count = page_count
        if chunk_count is not None:
            document.chunk_count = chunk_count
        self.session.add(document)
        await self.session.commit()
        await self.session.refresh(document)
        return document

    async def delete_by_ids(self, document_ids: list[UUID]) -> Sequence[Document]:
        documents = await self.get_by_ids(document_ids)
        if not documents:
            return []

        for document in documents:
            await self.session.delete(document)

        await self.session.commit()
        return documents
