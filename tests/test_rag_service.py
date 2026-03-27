from __future__ import annotations

from uuid import UUID

import pytest

pytest.importorskip("pydantic_settings")

from app.schemas.chat import ChatResponse, ChatTurn
from app.services.rag_service import NO_CONTEXT_ANSWER, RAGService
from app.services.types import RetrievalResult


class FakeCacheService:
    def __init__(self) -> None:
        self.store: dict[str, dict] = {}

    async def get_json(self, key: str):
        return self.store.get(key)

    async def set_json(self, key: str, value: dict, ttl_seconds: int | None = None):  # noqa: ARG002
        self.store[key] = value


class FakeRetrievalService:
    def __init__(self, rows: list[RetrievalResult]):
        self.rows = rows

    async def retrieve(self, *args, **kwargs):  # noqa: ANN002, ANN003
        return self.rows


class CaptureQueryRetrievalService(FakeRetrievalService):
    def __init__(self, rows: list[RetrievalResult]):
        super().__init__(rows)
        self.last_query: str = ""

    async def retrieve(self, *args, **kwargs):  # noqa: ANN002, ANN003
        self.last_query = str(args[0]) if args else ""
        return await super().retrieve(*args, **kwargs)


class FakeLLMService:
    async def generate(self, prompt: str) -> str:
        return "Grounded answer"

    async def stream_generate(self, prompt: str):  # noqa: ARG002
        yield "Grounded "
        yield "answer"


class FailOnGenerateLLMService(FakeLLMService):
    async def generate(self, prompt: str) -> str:  # noqa: ARG002
        raise AssertionError("LLM generate should not be called for deterministic NMT calculation")


class BadNumericLLMService(FakeLLMService):
    def __init__(self) -> None:
        self.calls = 0

    async def generate(self, prompt: str) -> str:  # noqa: ARG002
        self.calls += 1
        if self.calls == 1:
            return "3баллов"
        return "2 балла"


@pytest.mark.asyncio
async def test_rag_service_no_context_returns_fallback() -> None:
    rag_service = RAGService(
        retrieval_service=FakeRetrievalService([]),
        llm_service=FakeLLMService(),
        cache_service=FakeCacheService(),
    )

    response = await rag_service.answer(
        question="Unknown?",
        document_ids=None,
        top_k=5,
        use_hybrid_search=True,
        use_reranker=False,
    )

    assert isinstance(response, ChatResponse)
    assert response.answer == NO_CONTEXT_ANSWER
    assert response.sources == []


@pytest.mark.asyncio
async def test_rag_service_nmt_without_three_scores_returns_actionable_message() -> None:
    rag_service = RAGService(
        retrieval_service=FakeRetrievalService([]),
        llm_service=FakeLLMService(),
        cache_service=FakeCacheService(),
    )

    response = await rag_service.answer(
        question="Порахуй мені конкурсний бал по НМТ.",
        document_ids=None,
        top_k=5,
        use_hybrid_search=True,
        use_reranker=False,
    )

    assert response.answer.startswith("Щоб порахувати конкурсний бал НМТ")
    assert "165, 172, 181" in response.answer
    assert response.sources == []


@pytest.mark.asyncio
async def test_rag_service_nmt_with_three_scores_but_no_formula_requests_coefficients() -> None:
    rag_service = RAGService(
        retrieval_service=FakeRetrievalService([]),
        llm_service=FakeLLMService(),
        cache_service=FakeCacheService(),
    )

    response = await rag_service.answer(
        question="Порахуй конкурсний бал НМТ за 3 предметами: 171, 163, 152.",
        document_ids=None,
        top_k=5,
        use_hybrid_search=True,
        use_reranker=False,
    )

    assert response.answer.startswith("Маю 3 бали НМТ")
    assert "K1, K2, K3" in response.answer
    assert response.sources == []


@pytest.mark.asyncio
async def test_rag_service_nmt_average_formula_in_question_is_used() -> None:
    rag_service = RAGService(
        retrieval_service=FakeRetrievalService([]),
        llm_service=FailOnGenerateLLMService(),
        cache_service=FakeCacheService(),
    )

    response = await rag_service.answer(
        question="Конкурсний бал розраховується як середнє арифметичне трьох балів НМТ (171, 163, 152).",
        document_ids=None,
        top_k=5,
        use_hybrid_search=True,
        use_reranker=False,
    )

    assert "(171 + 163 + 152) / 3 = 162." in response.answer
    assert response.sources == []


@pytest.mark.asyncio
async def test_rag_service_returns_answer_with_sources() -> None:
    rows = [
        RetrievalResult(
            document_id=UUID("00000000-0000-0000-0000-000000000001"),
            filename="sample.pdf",
            page_number=3,
            chunk_index=0,
            text="Important factual context",
            score=0.91,
        )
    ]

    rag_service = RAGService(
        retrieval_service=FakeRetrievalService(rows),
        llm_service=FakeLLMService(),
        cache_service=FakeCacheService(),
    )

    response = await rag_service.answer(
        question="What is important?",
        document_ids=None,
        top_k=5,
        use_hybrid_search=True,
        use_reranker=False,
    )

    assert response.answer == "Grounded answer"
    assert len(response.sources) == 1
    assert response.sources[0].filename == "sample.pdf"


@pytest.mark.asyncio
async def test_rag_service_deterministic_nmt_score_without_llm_generation() -> None:
    rows = [
        RetrievalResult(
            document_id=UUID("00000000-0000-0000-0000-000000000001"),
            filename="rules.pdf",
            page_number=1,
            chunk_index=0,
            text="Формула конкурсного бала: К1 = 0,4; К2 = 0,35; К3 = 0,25.",
            score=0.95,
        )
    ]

    rag_service = RAGService(
        retrieval_service=FakeRetrievalService(rows),
        llm_service=FailOnGenerateLLMService(),
        cache_service=FakeCacheService(),
    )

    response = await rag_service.answer(
        question="Порахуй конкурсний бал за НМТ: 170, 160, 180.",
        document_ids=None,
        top_k=5,
        use_hybrid_search=True,
        use_reranker=False,
    )

    assert "170 x 0.4 + 160 x 0.35 + 180 x 0.25 = 169." in response.answer
    assert "НМТ" in response.answer
    assert len(response.sources) == 1


@pytest.mark.asyncio
async def test_rag_service_bad_numeric_answer_falls_back_after_retry() -> None:
    rows = [
        RetrievalResult(
            document_id=UUID("00000000-0000-0000-0000-000000000001"),
            filename="rules.pdf",
            page_number=1,
            chunk_index=0,
            text="Правила НМТ для вступу.",
            score=0.88,
        )
    ]
    llm = BadNumericLLMService()
    rag_service = RAGService(
        retrieval_service=FakeRetrievalService(rows),
        llm_service=llm,
        cache_service=FakeCacheService(),
    )

    response = await rag_service.answer(
        question="Порахуй конкурсний бал за НМТ: 170, 160, 180.",
        document_ids=None,
        top_k=5,
        use_hybrid_search=True,
        use_reranker=False,
    )

    assert response.answer.startswith("Маю 3 бали НМТ")
    assert "K1, K2, K3" in response.answer
    assert llm.calls == 0


@pytest.mark.asyncio
async def test_rag_service_nmt_without_three_scores_skips_llm_even_with_context() -> None:
    rows = [
        RetrievalResult(
            document_id=UUID("00000000-0000-0000-0000-000000000001"),
            filename="rules.pdf",
            page_number=1,
            chunk_index=0,
            text="Формула конкурсного бала: К1 = 0,4; К2 = 0,35; К3 = 0,25.",
            score=0.95,
        )
    ]
    rag_service = RAGService(
        retrieval_service=FakeRetrievalService(rows),
        llm_service=FailOnGenerateLLMService(),
        cache_service=FakeCacheService(),
    )

    response = await rag_service.answer(
        question="Порахуй конкурсний бал за НМТ.",
        document_ids=None,
        top_k=5,
        use_hybrid_search=True,
        use_reranker=False,
    )

    assert response.answer.startswith("Щоб порахувати конкурсний бал НМТ")


@pytest.mark.asyncio
async def test_rag_service_inline_coefficients_in_question_are_used() -> None:
    rag_service = RAGService(
        retrieval_service=FakeRetrievalService([]),
        llm_service=FailOnGenerateLLMService(),
        cache_service=FakeCacheService(),
    )

    response = await rag_service.answer(
        question="Порахуй конкурсний бал НМТ: 171, 163, 152, К1=0,4 К2=0,35 К3=0,25.",
        document_ids=None,
        top_k=5,
        use_hybrid_search=True,
        use_reranker=False,
    )

    assert "171 x 0.4 + 163 x 0.35 + 152 x 0.25 = 163.45." in response.answer


@pytest.mark.asyncio
async def test_rag_service_uses_history_in_retrieval_query() -> None:
    rows = [
        RetrievalResult(
            document_id=UUID("00000000-0000-0000-0000-000000000001"),
            filename="sample.pdf",
            page_number=2,
            chunk_index=3,
            text="Section 2 explains admission requirements.",
            score=0.91,
        )
    ]
    retrieval = CaptureQueryRetrievalService(rows)
    rag_service = RAGService(
        retrieval_service=retrieval,
        llm_service=FakeLLMService(),
        cache_service=FakeCacheService(),
    )

    response = await rag_service.answer(
        question="What about it?",
        document_ids=None,
        top_k=5,
        use_hybrid_search=True,
        use_reranker=False,
        history=[
            ChatTurn(role="user", content="Tell me about section 2."),
            ChatTurn(role="assistant", content="Section 2 is about admission requirements."),
        ],
    )

    assert "Conversation context:" in retrieval.last_query
    assert "Tell me about section 2." in retrieval.last_query
    assert response.answer == "Grounded answer"
