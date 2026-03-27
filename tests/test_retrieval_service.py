from __future__ import annotations

from uuid import UUID

import pytest

pytest.importorskip("pydantic_settings")

from app.repositories.chunk_repository import RetrievedChunkRecord
from app.services.retrieval_service import RetrievalService, settings as retrieval_settings


class FakeChunkRepository:
    def __init__(self, rows: list[RetrievedChunkRecord]) -> None:
        self._rows = rows

    async def search(  # noqa: ANN001
        self,
        query_embedding,
        query_text,
        top_k,
        document_ids,
        use_hybrid_search,
    ) -> list[RetrievedChunkRecord]:
        return self._rows


class FakeEmbeddingService:
    async def embed_query(self, query: str) -> list[float]:  # noqa: ARG002
        return [0.1, 0.2, 0.3]


@pytest.mark.asyncio
async def test_retrieve_backfills_when_threshold_filters_everything() -> None:
    rows = [
        RetrievedChunkRecord(
            id=UUID("00000000-0000-0000-0000-000000000011"),
            document_id=UUID("00000000-0000-0000-0000-000000000001"),
            filename="a.pdf",
            page_number=1,
            chunk_index=0,
            text="alpha",
            token_count=10,
            score=0.09,
            vector_score=0.09,
            bm25_score=0.0,
        ),
        RetrievedChunkRecord(
            id=UUID("00000000-0000-0000-0000-000000000012"),
            document_id=UUID("00000000-0000-0000-0000-000000000001"),
            filename="a.pdf",
            page_number=2,
            chunk_index=1,
            text="beta",
            token_count=11,
            score=0.08,
            vector_score=0.08,
            bm25_score=0.0,
        ),
        RetrievedChunkRecord(
            id=UUID("00000000-0000-0000-0000-000000000013"),
            document_id=UUID("00000000-0000-0000-0000-000000000001"),
            filename="a.pdf",
            page_number=3,
            chunk_index=2,
            text="gamma",
            token_count=12,
            score=0.07,
            vector_score=0.07,
            bm25_score=0.0,
        ),
    ]
    service = RetrievalService(
        chunk_repository=FakeChunkRepository(rows),  # type: ignore[arg-type]
        embedding_service=FakeEmbeddingService(),  # type: ignore[arg-type]
    )

    result = await service.retrieve(
        "test query",
        top_k=5,
        document_ids=None,
        use_hybrid_search=True,
        use_reranker=False,
    )

    assert len(result) == min(len(rows), retrieval_settings.retrieval_min_results)
    assert [item.chunk_index for item in result] == [0, 1, 2][: len(result)]
