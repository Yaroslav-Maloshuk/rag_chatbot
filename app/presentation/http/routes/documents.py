from __future__ import annotations

import uuid
from pathlib import Path

import aiofiles
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.db.models import DocumentStatus
from app.presentation.http.deps import get_ingestion_service, get_session
from app.repositories.document_repository import DocumentRepository
from app.schemas.document import (
    DocumentDeleteRequest,
    DocumentDeleteResponse,
    DocumentListResponse,
    DocumentStatusResponse,
    DocumentUploadResponse,
)
from app.services.ingestion_service import IngestionService
from app.tasks.document_tasks import process_document_task

settings = get_settings()

router = APIRouter(prefix="/documents", tags=["documents"])


async def _save_upload_to_disk(upload_file: UploadFile, destination: Path) -> int:
    max_size_bytes = settings.max_upload_size_mb * 1024 * 1024
    chunk_size = 1024 * 1024
    total_size = 0

    try:
        async with aiofiles.open(destination, "wb") as output:
            while chunk := await upload_file.read(chunk_size):
                total_size += len(chunk)
                if total_size > max_size_bytes:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"File too large. Max size is {settings.max_upload_size_mb} MB",
                    )
                await output.write(chunk)
    except Exception:
        if destination.exists():
            destination.unlink()
        raise

    await upload_file.close()
    return total_size


@router.post("/upload", response_model=DocumentUploadResponse, status_code=status.HTTP_202_ACCEPTED)
async def upload_pdf(
    file: UploadFile = File(...),
    process_sync: bool = Query(default=True),
    session: AsyncSession = Depends(get_session),
    ingestion_service: IngestionService = Depends(get_ingestion_service),
) -> DocumentUploadResponse:
    filename = file.filename or "document.pdf"
    content_type = (file.content_type or "").lower()
    if "pdf" not in content_type and not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Only PDF files are supported")

    settings.upload_path.mkdir(parents=True, exist_ok=True)

    source_name = Path(filename).name
    unique_name = f"{uuid.uuid4()}_{source_name}"
    destination = settings.upload_path / unique_name

    file_size = await _save_upload_to_disk(file, destination)

    document_repo = DocumentRepository(session)
    document = await document_repo.create(
        filename=source_name,
        file_path=str(destination),
        file_size_bytes=file_size,
    )

    file_size_mb = file_size / (1024 * 1024)
    is_large_file = file_size_mb >= settings.large_file_threshold_mb

    if process_sync and not is_large_file:
        await ingestion_service.ingest_document(document.id)
        refreshed = await document_repo.get(document.id)
        return DocumentUploadResponse(
            document_id=document.id,
            filename=document.filename,
            file_size_bytes=file_size,
            status=refreshed.status if refreshed else DocumentStatus.UPLOADED,
            message="PDF uploaded and processed synchronously",
        )

    task = process_document_task.delay(str(document.id))
    return DocumentUploadResponse(
        document_id=document.id,
        filename=document.filename,
        file_size_bytes=file_size,
        status=DocumentStatus.UPLOADED,
        task_id=task.id,
        message="PDF uploaded successfully and queued for async ingestion",
    )


@router.get("/{document_id}", response_model=DocumentStatusResponse)
async def get_document(document_id: uuid.UUID, session: AsyncSession = Depends(get_session)) -> DocumentStatusResponse:
    document_repo = DocumentRepository(session)
    document = await document_repo.get(document_id)
    if document is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    return DocumentStatusResponse.model_validate(document)


@router.get("", response_model=DocumentListResponse)
async def list_documents(
    limit: int = Query(default=100, ge=1, le=500),
    session: AsyncSession = Depends(get_session),
) -> DocumentListResponse:
    document_repo = DocumentRepository(session)
    rows = await document_repo.list(limit=limit)
    return DocumentListResponse(items=[DocumentStatusResponse.model_validate(row) for row in rows])


@router.post("/delete", response_model=DocumentDeleteResponse)
async def delete_documents(
    payload: DocumentDeleteRequest,
    session: AsyncSession = Depends(get_session),
) -> DocumentDeleteResponse:
    document_repo = DocumentRepository(session)
    documents = await document_repo.get_by_ids(payload.document_ids)

    found_ids = {doc.id for doc in documents}
    missing_ids = [str(doc_id) for doc_id in payload.document_ids if doc_id not in found_ids]
    if missing_ids:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Documents not found: {', '.join(missing_ids)}",
        )

    not_ready = [doc for doc in documents if doc.status != DocumentStatus.READY]
    if not_ready:
        not_ready_text = ", ".join(f"{doc.filename} ({doc.status.value})" for doc in not_ready)
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Only ready documents can be deleted: {not_ready_text}",
        )

    deleted_documents = await document_repo.delete_by_ids(payload.document_ids)
    for document in deleted_documents:
        path = Path(document.file_path)
        if path.exists():
            path.unlink()

    return DocumentDeleteResponse(
        deleted_ids=[doc.id for doc in deleted_documents],
        deleted_count=len(deleted_documents),
    )

