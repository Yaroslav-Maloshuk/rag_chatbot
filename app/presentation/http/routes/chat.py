from __future__ import annotations

import json
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.db.models import DocumentStatus
from app.presentation.http.deps import get_rag_service, get_session
from app.repositories.document_repository import DocumentRepository
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.rag_service import RAGService

settings = get_settings()

router = APIRouter(prefix="/chat", tags=["chat"])


async def _ensure_documents_ready(
    session: AsyncSession,
    document_ids: list[UUID] | None,
) -> None:
    if not document_ids:
        return

    document_repository = DocumentRepository(session)
    documents = await document_repository.get_by_ids(document_ids)
    found_ids = {doc.id for doc in documents}
    missing_ids = [str(doc_id) for doc_id in document_ids if doc_id not in found_ids]
    if missing_ids:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document(s) not found: {', '.join(missing_ids)}",
        )

    not_ready = [doc for doc in documents if doc.status != DocumentStatus.READY]
    if not_ready:
        not_ready_text = ", ".join(f"{doc.filename} ({doc.status.value})" for doc in not_ready)
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Selected documents are not ready for chat: {not_ready_text}",
        )


@router.post("", response_model=ChatResponse)
async def chat(
    payload: ChatRequest,
    rag_service: RAGService = Depends(get_rag_service),
    session: AsyncSession = Depends(get_session),
) -> ChatResponse:
    await _ensure_documents_ready(session, payload.document_ids)
    return await rag_service.answer(
        question=payload.question,
        document_ids=payload.document_ids,
        top_k=payload.top_k or settings.retrieval_top_k,
        use_hybrid_search=(
            payload.use_hybrid_search
            if payload.use_hybrid_search is not None
            else settings.enable_hybrid_search
        ),
        use_reranker=(
            payload.use_reranker
            if payload.use_reranker is not None
            else settings.enable_reranker
        ),
        history=payload.history,
    )


@router.post("/stream")
async def stream_chat(
    payload: ChatRequest,
    rag_service: RAGService = Depends(get_rag_service),
    session: AsyncSession = Depends(get_session),
) -> StreamingResponse:
    await _ensure_documents_ready(session, payload.document_ids)

    async def event_generator():
        async for item in rag_service.stream_answer(
            question=payload.question,
            document_ids=payload.document_ids,
            top_k=payload.top_k or settings.retrieval_top_k,
            use_hybrid_search=(
                payload.use_hybrid_search
                if payload.use_hybrid_search is not None
                else settings.enable_hybrid_search
            ),
            use_reranker=(
                payload.use_reranker
                if payload.use_reranker is not None
                else settings.enable_reranker
            ),
            history=payload.history,
        ):
            yield f"event: {item['event']}\ndata: {json.dumps(item['data'])}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

