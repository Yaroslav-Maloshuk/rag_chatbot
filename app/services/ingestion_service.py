from __future__ import annotations

import logging
from uuid import UUID

from app.db.models import DocumentStatus
from app.repositories.chunk_repository import ChunkRepository
from app.repositories.document_repository import DocumentRepository
from app.services.chunking_service import ChunkingService
from app.services.embedding_service import EmbeddingService
from app.services.pdf_service import PDFExtractionError, PDFService

logger = logging.getLogger(__name__)


class IngestionService:
    def __init__(
        self,
        *,
        document_repository: DocumentRepository,
        chunk_repository: ChunkRepository,
        pdf_service: PDFService,
        chunking_service: ChunkingService,
        embedding_service: EmbeddingService | None = None,
    ) -> None:
        self.document_repository = document_repository
        self.chunk_repository = chunk_repository
        self.pdf_service = pdf_service
        self.chunking_service = chunking_service
        self._embedding_service = embedding_service

    def _get_embedding_service(self) -> EmbeddingService:
        # Lazy initialization keeps status transitions fast and avoids model-loading
        # overhead before the document is marked as processing.
        if self._embedding_service is None:
            self._embedding_service = EmbeddingService()
        return self._embedding_service

    async def ingest_document(self, document_id: UUID) -> None:
        document = await self.document_repository.get(document_id)
        if document is None:
            logger.warning("Document %s not found for ingestion", document_id)
            return

        await self.document_repository.update_status(document, DocumentStatus.PROCESSING, error_message=None)

        total_pages = 0
        produced_chunks = 0

        try:
            pages, total_pages = self.pdf_service.extract_pages(document.file_path)
            if not pages:
                raise PDFExtractionError(
                    "No extractable text found in PDF. If this is a scanned document, OCR may be required."
                )

            chunks = self.chunking_service.chunk_pages(pages)
            if not chunks:
                raise PDFExtractionError("No chunks produced from extracted PDF text")
            produced_chunks = len(chunks)

            embeddings = await self._get_embedding_service().embed_texts([chunk.text for chunk in chunks])

            rows = [
                {
                    "document_id": document.id,
                    "page_number": chunk.page_number,
                    "chunk_index": chunk.chunk_index,
                    "text": chunk.text,
                    "token_count": chunk.token_count,
                    "embedding": embedding,
                }
                for chunk, embedding in zip(chunks, embeddings, strict=True)
            ]

            await self.chunk_repository.delete_by_document(document.id)
            await self.chunk_repository.bulk_insert(rows)

            await self.document_repository.update_status(
                document,
                DocumentStatus.READY,
                error_message=None,
                page_count=total_pages,
                chunk_count=produced_chunks,
            )
            logger.info("Document %s ingested successfully (%s chunks)", document.id, produced_chunks)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Ingestion failed for %s", document.id)
            await self.document_repository.update_status(
                document,
                DocumentStatus.FAILED,
                error_message=str(exc),
                page_count=total_pages,
                chunk_count=produced_chunks,
            )
