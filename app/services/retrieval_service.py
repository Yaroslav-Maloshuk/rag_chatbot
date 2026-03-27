from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING
from uuid import UUID

from app.core.config import get_settings
from app.core.runtime_device import get_runtime_device
from app.repositories.chunk_repository import ChunkRepository
from app.services.embedding_service import EmbeddingService
from app.services.types import RetrievalResult

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder

settings = get_settings()


class RetrievalService:
    def __init__(self, chunk_repository: ChunkRepository, embedding_service: EmbeddingService) -> None:
        self._chunk_repository = chunk_repository
        self._embedding_service = embedding_service
        self._reranker: CrossEncoder | None = None

    @staticmethod
    def _result_key(row: RetrievalResult) -> tuple[UUID, int, int]:
        return row.document_id, row.page_number, row.chunk_index

    def _ensure_minimum_results(
        self,
        filtered_results: list[RetrievalResult],
        candidate_results: list[RetrievalResult],
    ) -> list[RetrievalResult]:
        minimum = max(0, settings.retrieval_min_results)
        if minimum == 0 or len(filtered_results) >= minimum:
            return filtered_results

        merged = list(filtered_results)
        seen = {self._result_key(row) for row in merged}
        for row in candidate_results:
            row_key = self._result_key(row)
            if row_key in seen:
                continue
            merged.append(row)
            seen.add(row_key)
            if len(merged) >= minimum:
                break
        return merged

    def _get_reranker(self):
        if self._reranker is None and settings.enable_reranker:
            from sentence_transformers import CrossEncoder

            runtime = get_runtime_device()
            try:
                self._reranker = CrossEncoder(
                    settings.reranker_model_name,
                    device=runtime.sentence_transformers_device,
                )
            except Exception:  # noqa: BLE001
                self._reranker = CrossEncoder(settings.reranker_model_name, device="cpu")
        return self._reranker

    async def _rerank(
        self,
        query: str,
        rows: list[RetrievalResult],
        enabled: bool,
    ) -> list[RetrievalResult]:
        if not enabled or not settings.enable_reranker or not rows:
            return rows

        reranker = self._get_reranker()
        if reranker is None:
            return rows

        pairs = [[query, row.text] for row in rows]

        def _predict() -> list[float]:
            return reranker.predict(pairs).tolist()

        scores = await asyncio.to_thread(_predict)

        reranked: list[RetrievalResult] = []
        for row, score in zip(rows, scores, strict=True):
            reranked.append(
                RetrievalResult(
                    document_id=row.document_id,
                    filename=row.filename,
                    page_number=row.page_number,
                    chunk_index=row.chunk_index,
                    text=row.text,
                    score=float(score),
                )
            )

        reranked.sort(key=lambda item: item.score, reverse=True)
        return reranked

    async def retrieve(
        self,
        query: str,
        *,
        top_k: int,
        document_ids: list[UUID] | None,
        use_hybrid_search: bool,
        use_reranker: bool,
    ) -> list[RetrievalResult]:
        query_embedding = await self._embedding_service.embed_query(query)

        candidate_rows = await self._chunk_repository.search(
            query_embedding=query_embedding,
            query_text=query,
            top_k=max(top_k, settings.retrieval_candidate_k),
            document_ids=document_ids,
            use_hybrid_search=use_hybrid_search,
        )

        candidate_results = [
            RetrievalResult(
                document_id=row.document_id,
                filename=row.filename,
                page_number=row.page_number,
                chunk_index=row.chunk_index,
                text=row.text,
                score=row.score,
            )
            for row in candidate_rows
        ]

        filtered_results = [row for row in candidate_results if row.score >= settings.min_relevance_score]

        # If threshold is too strict for a specific query/language mix, keep a minimum pool
        # so the model still gets evidence to reason over.
        if not filtered_results and use_reranker and settings.enable_reranker:
            prepared_results = candidate_results
        else:
            prepared_results = self._ensure_minimum_results(filtered_results, candidate_results)

        reranked = await self._rerank(query, prepared_results, enabled=use_reranker)
        return reranked[:top_k]
