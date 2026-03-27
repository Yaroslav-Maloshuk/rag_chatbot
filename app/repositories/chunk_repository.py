from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import delete, func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.db.models import Chunk, Document, DocumentStatus

settings = get_settings()


@dataclass(slots=True)
class RetrievedChunkRecord:
    id: UUID
    document_id: UUID
    filename: str
    page_number: int
    chunk_index: int
    text: str
    token_count: int
    score: float
    vector_score: float
    bm25_score: float


class ChunkRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def delete_by_document(self, document_id: UUID) -> None:
        await self.session.execute(delete(Chunk).where(Chunk.document_id == document_id))
        await self.session.commit()

    async def bulk_insert(self, rows: Sequence[dict]) -> None:
        if not rows:
            return
        stmt = insert(Chunk).values(list(rows))
        await self.session.execute(stmt)
        await self.session.commit()

    async def search(
        self,
        *,
        query_embedding: list[float],
        query_text: str,
        top_k: int,
        document_ids: list[UUID] | None,
        use_hybrid_search: bool,
    ) -> list[RetrievedChunkRecord]:
        vector_score = (1 - Chunk.embedding.cosine_distance(query_embedding)).label("vector_score")
        bm25_score = func.ts_rank_cd(
            func.to_tsvector(settings.text_search_config, Chunk.text),
            func.plainto_tsquery(settings.text_search_config, query_text),
        ).label("bm25_score")

        if use_hybrid_search:
            score = (
                settings.hybrid_vector_weight * vector_score
                + settings.hybrid_bm25_weight * bm25_score
            ).label("score")
        else:
            score = vector_score.label("score")

        stmt = (
            select(
                Chunk.id,
                Chunk.document_id,
                Document.filename,
                Chunk.page_number,
                Chunk.chunk_index,
                Chunk.text,
                Chunk.token_count,
                score,
                vector_score,
                bm25_score,
            )
            .join(Document, Chunk.document_id == Document.id)
            .where(Document.status == DocumentStatus.READY)
        )

        if document_ids:
            stmt = stmt.where(Chunk.document_id.in_(document_ids))

        if use_hybrid_search:
            stmt = stmt.order_by(score.desc())
        else:
            stmt = stmt.order_by(vector_score.desc())

        stmt = stmt.limit(top_k)

        result = await self.session.execute(stmt)

        rows: list[RetrievedChunkRecord] = []
        for row in result.all():
            rows.append(
                RetrievedChunkRecord(
                    id=row.id,
                    document_id=row.document_id,
                    filename=row.filename,
                    page_number=row.page_number,
                    chunk_index=row.chunk_index,
                    text=row.text,
                    token_count=row.token_count,
                    score=float(row.score or 0.0),
                    vector_score=float(row.vector_score or 0.0),
                    bm25_score=float(row.bm25_score or 0.0),
                )
            )
        return rows
