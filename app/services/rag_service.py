from __future__ import annotations

import hashlib
import json
import re
from collections.abc import AsyncGenerator
from uuid import UUID

from app.core.config import get_settings
from app.schemas.chat import ChatResponse, ChatTurn, SourceChunk
from app.services.cache_service import CacheService
from app.services.llm_service import LLMService
from app.services.retrieval_service import RetrievalService
from app.services.types import RetrievalResult
from app.utils.prompting import build_calc_rag_prompt, build_rag_prompt, build_refine_rag_prompt

settings = get_settings()

NO_CONTEXT_ANSWER = "I don't know based on the provided documents."
CALC_KEYWORDS = (
    "calculate",
    "calculation",
    "score",
    "formula",
    "конкурс",
    "конкурсн",
    "бал",
    "балл",
    "нмт",
    "nmt",
    "вступ",
    "розрах",
    "обчисл",
    "порах",
    "скільки",
    "сколько",
)
LETTER_BOUNDARY_RE = re.compile(r"(\d)([A-Za-zА-Яа-яЁёЇїІіЄєҐґ])|([A-Za-zА-Яа-яЁёЇїІіЄєҐґ])(\d)")
NUMBER_RE = re.compile(r"(?<!\d)(\d{1,4}(?:[.,]\d+)?)(?!\d)")
CYRILLIC_RE = re.compile(r"[А-Яа-яЁёЇїІіЄєҐґ]")
UKRAINIAN_LETTERS_RE = re.compile(r"[ЇїІіЄєҐґ]")
UKRAINIAN_HINT_TERMS = ("порахуй", "конкурсний", "предметів", "виш", "вступ")
RUSSIAN_HINT_TERMS = ("посчитай", "конкурсный", "предметам", "вуз", "поступ")
NMT_TERMS = ("нмт", "nmt")
ADMISSION_TERMS = ("конкурс", "вступ", "внз", "виш", "універс", "admission", "university")
AVERAGE_FORMULA_TERMS = (
    "середнє арифметичне",
    "среднее арифметическое",
    "arithmetic mean",
    "average",
)
MAX_HISTORY_TURNS = 12
MAX_HISTORY_CHARS = 2200


class RAGService:
    def __init__(
        self,
        *,
        retrieval_service: RetrievalService,
        llm_service: LLMService,
        cache_service: CacheService,
    ) -> None:
        self._retrieval_service = retrieval_service
        self._llm_service = llm_service
        self._cache_service = cache_service

    @staticmethod
    def _build_cache_key(
        question: str,
        document_ids: list[UUID] | None,
        top_k: int,
        use_hybrid_search: bool,
        use_reranker: bool,
        history_signature: str,
    ) -> str:
        payload = {
            "question": question,
            "document_ids": sorted([str(doc_id) for doc_id in document_ids or []]),
            "top_k": top_k,
            "use_hybrid_search": use_hybrid_search,
            "use_reranker": use_reranker,
            "history_signature": history_signature,
            "embedding_model": settings.embedding_model_name,
            "llm_model": settings.llm_model_name,
            "text_search_config": settings.text_search_config,
            "prompt_version": 10,
        }
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
        return f"rag:chat:{digest}"

    @staticmethod
    def _normalize_history_content(content: str) -> str:
        return re.sub(r"\s+", " ", content.strip())

    @staticmethod
    def _build_history_text(history: list[ChatTurn] | None) -> str:
        if not history:
            return ""

        lines: list[str] = []
        for turn in history[-MAX_HISTORY_TURNS:]:
            content = RAGService._normalize_history_content(turn.content)
            if not content:
                continue
            speaker = "User" if turn.role == "user" else "Assistant"
            lines.append(f"{speaker}: {content}")

        history_text = "\n".join(lines).strip()
        if len(history_text) > MAX_HISTORY_CHARS:
            history_text = history_text[-MAX_HISTORY_CHARS:]
        return history_text

    @staticmethod
    def _build_retrieval_query(question: str, history_text: str) -> str:
        if not history_text:
            return question
        return (
            "Conversation context:\n"
            f"{history_text}\n\n"
            "Current user question:\n"
            f"{question.strip()}"
        )

    def _build_context(self, chunks: list[RetrievalResult]) -> str:
        budget = settings.max_context_tokens
        consumed = 0
        context_parts: list[str] = []

        for chunk in chunks:
            chunk_tokens = max(1, len(chunk.text.split()))
            if consumed + chunk_tokens > budget:
                break
            consumed += chunk_tokens
            context_parts.append(
                f"[Source: {chunk.filename} | page {chunk.page_number} | chunk {chunk.chunk_index}]\n{chunk.text}"
            )

        return "\n\n".join(context_parts)

    @staticmethod
    def _build_sources(chunks: list[RetrievalResult]) -> list[SourceChunk]:
        return [
            SourceChunk(
                document_id=chunk.document_id,
                filename=chunk.filename,
                page_number=chunk.page_number,
                chunk_index=chunk.chunk_index,
                score=round(chunk.score, 5),
                text_preview=(chunk.text[:220] + "...") if len(chunk.text) > 220 else chunk.text,
            )
            for chunk in chunks
        ]

    @staticmethod
    def _is_calculation_question(question: str) -> bool:
        lowered = question.lower()
        if any(keyword in lowered for keyword in CALC_KEYWORDS):
            return True
        return bool(re.search(r"\d+\s*[-+*/]", lowered))

    @staticmethod
    def _extract_numbers(text: str) -> list[float]:
        values: list[float] = []
        for match in NUMBER_RE.findall(text):
            try:
                values.append(float(match.replace(",", ".")))
            except ValueError:
                continue
        return values

    @staticmethod
    def _extract_nmt_subject_scores(question: str) -> list[float]:
        return [value for value in RAGService._extract_numbers(question) if 80 <= value <= 200][:3]

    @staticmethod
    def _format_number(value: float) -> str:
        return f"{value:.3f}".rstrip("0").rstrip(".")

    @staticmethod
    def _detect_output_language(question: str) -> str:
        lowered = question.lower()
        if any(term in lowered for term in UKRAINIAN_HINT_TERMS):
            return "uk"
        if any(term in lowered for term in RUSSIAN_HINT_TERMS):
            return "ru"
        if UKRAINIAN_LETTERS_RE.search(question):
            return "uk"
        if CYRILLIC_RE.search(question):
            return "ru"
        return "en"

    @staticmethod
    def _extract_first_float(patterns: tuple[str, ...], text: str) -> float | None:
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if not match:
                continue
            raw = match.group(1).replace(",", ".")
            try:
                value = float(raw)
            except ValueError:
                continue
            if 0 < value <= 2:
                return value
        return None

    @staticmethod
    def _has_average_formula_intent(text: str) -> bool:
        lowered = text.lower()
        return any(term in lowered for term in AVERAGE_FORMULA_TERMS)

    @staticmethod
    def _extract_k123_coefficients(text: str) -> tuple[float, float, float] | None:
        coefficients: list[float] = []
        for idx in (1, 2, 3):
            value = RAGService._extract_first_float(
                (
                    rf"\b[кk]\s*{idx}\b[^0-9]{{0,8}}([01](?:[.,]\d+)?)",
                    rf"\bкоеф(?:іцієнт|ициент)?\s*{idx}\b[^0-9]{{0,8}}([01](?:[.,]\d+)?)",
                ),
                text,
            )
            if value is None:
                return None
            coefficients.append(value)
        return coefficients[0], coefficients[1], coefficients[2]

    @staticmethod
    def _has_extended_admission_formula(context: str) -> bool:
        return bool(re.search(r"\b[кk]\s*[45]\b", context, flags=re.IGNORECASE))

    def _build_deterministic_nmt_answer(self, *, question: str, context: str) -> str | None:
        if not self._has_admission_nmt_intent(question):
            return None
        if self._has_extended_admission_formula(context):
            return None

        scores = self._extract_nmt_subject_scores(question)
        if len(scores) < 3:
            return None

        coefficients = self._extract_k123_coefficients(context)
        if coefficients is None:
            coefficients = self._extract_k123_coefficients(question)
        if coefficients is None and self._has_average_formula_intent(f"{question}\n{context}"):
            result = sum(scores) / 3
            s1, s2, s3 = (self._format_number(value) for value in scores)
            final_value = self._format_number(result)

            language = self._detect_output_language(question)
            if language == "uk":
                return (
                    "Конкурсний бал (середнє арифметичне 3 балів НМТ) = "
                    f"({s1} + {s2} + {s3}) / 3 = {final_value}."
                )
            if language == "ru":
                return (
                    "Конкурсный балл (среднее арифметическое 3 баллов НМТ) = "
                    f"({s1} + {s2} + {s3}) / 3 = {final_value}."
                )
            return (
                "Competitive score (arithmetic mean of 3 NMT scores) = "
                f"({s1} + {s2} + {s3}) / 3 = {final_value}."
            )
        if coefficients is None:
            return None

        result = scores[0] * coefficients[0] + scores[1] * coefficients[1] + scores[2] * coefficients[2]
        s1, s2, s3 = (self._format_number(value) for value in scores)
        k1, k2, k3 = (self._format_number(value) for value in coefficients)
        final_value = self._format_number(result)

        language = self._detect_output_language(question)
        if language == "uk":
            return (
                "Конкурсний бал (за 3 предметами НМТ) = "
                f"{s1} x {k1} + {s2} x {k2} + {s3} x {k3} = {final_value}."
            )
        if language == "ru":
            return (
                "Конкурсный балл (по 3 предметам НМТ) = "
                f"{s1} x {k1} + {s2} x {k2} + {s3} x {k3} = {final_value}."
            )
        return (
            "Competitive score (3 NMT subjects) = "
            f"{s1} x {k1} + {s2} x {k2} + {s3} x {k3} = {final_value}."
        )

    def _build_missing_nmt_scores_message(self, question: str) -> str | None:
        if not self._has_admission_nmt_intent(question):
            return None
        if len(self._extract_nmt_subject_scores(question)) >= 3:
            return None

        language = self._detect_output_language(question)
        if language == "uk":
            return (
                "Щоб порахувати конкурсний бал НМТ, надай 3 бали з предметів "
                "(наприклад: 165, 172, 181)."
            )
        if language == "ru":
            return (
                "Чтобы посчитать конкурсный балл НМТ, укажи 3 балла по предметам "
                "(например: 165, 172, 181)."
            )
        return "To calculate the NMT competitive score, provide 3 subject scores (for example: 165, 172, 181)."

    def _build_missing_nmt_formula_message(self, question: str) -> str | None:
        if not self._has_admission_nmt_intent(question):
            return None
        if len(self._extract_nmt_subject_scores(question)) < 3:
            return None
        if self._has_average_formula_intent(question):
            return None

        language = self._detect_output_language(question)
        if language == "uk":
            return (
                "Маю 3 бали НМТ, але для розрахунку не вистачає формули/коефіцієнтів "
                "K1, K2, K3. Додай коефіцієнти в запит або завантаж документ з правилами вступу."
            )
        if language == "ru":
            return (
                "У меня есть 3 балла НМТ, но для расчета не хватает формулы/коэффициентов "
                "K1, K2, K3. Добавь коэффициенты в запрос или загрузи документ с правилами поступления."
            )
        return (
            "I have 3 NMT scores, but calculation requires the formula/coefficients "
            "K1, K2, K3. Add coefficients to your question or upload an admission-rules document."
        )

    @staticmethod
    def _normalize_answer_text(answer: str) -> str:
        normalized = answer.strip()

        # Fix merged patterns like "3баллов" -> "3 баллов".
        def _split_joined_tokens(match: re.Match[str]) -> str:
            if match.group(1) and match.group(2):
                return f"{match.group(1)} {match.group(2)}"
            return f"{match.group(3)} {match.group(4)}"

        normalized = LETTER_BOUNDARY_RE.sub(_split_joined_tokens, normalized)
        normalized = re.sub(r"[ \t]{2,}", " ", normalized)
        return normalized.strip()

    def _looks_suspicious_calculation_answer(self, *, question: str, context: str, answer: str) -> bool:
        if not answer or answer == NO_CONTEXT_ANSWER:
            return False

        numbers = self._extract_numbers(answer)
        if not numbers:
            return True

        if re.search(r"\d[^\s\d]", answer):
            return True

        lowered = question.lower()
        is_admission_score_question = any(token in lowered for token in ("нмт", "nmt", "конкурс", "вступ"))
        if is_admission_score_question:
            max_answer_value = max(numbers)
            nmt_scores_in_question = self._extract_nmt_subject_scores(question)
            if len(nmt_scores_in_question) >= 3 and max_answer_value < 100:
                return True
            if max_answer_value < 50:
                context_numbers = self._extract_numbers(context)
                high_score_hints = [value for value in context_numbers if 80 <= value <= 200]
                if len(high_score_hints) >= 2:
                    return True

        return False

    @staticmethod
    def _has_admission_nmt_intent(question: str) -> bool:
        lowered = question.lower()
        return any(term in lowered for term in NMT_TERMS) or any(term in lowered for term in ADMISSION_TERMS)

    @staticmethod
    def _context_matches_admission_nmt(question: str, context: str) -> bool:
        if not RAGService._has_admission_nmt_intent(question):
            return True

        lowered_question = question.lower()
        lowered_context = context.lower()
        has_k123_formula = all(re.search(rf"\b[кk]\s*{idx}\b", lowered_context) for idx in (1, 2, 3))
        has_admission_formula_terms = has_k123_formula and any(
            term in lowered_context for term in ("конкурс", "вступ", "competitive", "admission")
        )
        question_has_nmt_terms = any(term in lowered_question for term in NMT_TERMS)
        if question_has_nmt_terms and not any(term in lowered_context for term in NMT_TERMS):
            if not has_admission_formula_terms:
                return False

        question_has_admission_terms = any(term in lowered_question for term in ADMISSION_TERMS)
        if question_has_admission_terms and not question_has_nmt_terms:
            if not any(term in lowered_context for term in ADMISSION_TERMS):
                return False

        has_nmt_terms = any(term in lowered_context for term in NMT_TERMS)
        has_admission_terms = any(term in lowered_context for term in ADMISSION_TERMS)
        return has_nmt_terms or has_admission_terms or has_admission_formula_terms

    async def _generate_answer_text(self, *, question: str, context: str, history_text: str = "") -> str:
        question_for_reasoning = f"{history_text}\n{question}".strip() if history_text else question

        if self._is_calculation_question(question_for_reasoning):
            missing_scores_message = self._build_missing_nmt_scores_message(question_for_reasoning)
            if missing_scores_message:
                return missing_scores_message

            deterministic_answer = self._build_deterministic_nmt_answer(
                question=question_for_reasoning,
                context=context,
            )
            if deterministic_answer:
                answer = deterministic_answer
            else:
                missing_formula_message = self._build_missing_nmt_formula_message(question_for_reasoning)
                if not self._context_matches_admission_nmt(question_for_reasoning, context):
                    return missing_formula_message or NO_CONTEXT_ANSWER

                has_inline_coefficients = self._extract_k123_coefficients(question_for_reasoning) is not None
                has_context_coefficients = self._extract_k123_coefficients(context) is not None
                if missing_formula_message and not has_inline_coefficients and not has_context_coefficients:
                    return missing_formula_message

                calc_prompt = build_calc_rag_prompt(
                    question=question,
                    context=context,
                    initial_answer="",
                    conversation_history=history_text,
                )
                answer = (await self._llm_service.generate(calc_prompt)).strip() or ""

                if self._looks_suspicious_calculation_answer(
                    question=question_for_reasoning,
                    context=context,
                    answer=answer,
                ):
                    retry_prompt = build_calc_rag_prompt(
                        question=question,
                        context=context,
                        initial_answer=(
                            f"{answer}\n"
                            "The draft above may be numerically inconsistent. Recalculate carefully and output only the corrected final answer."
                        ),
                        conversation_history=history_text,
                    )
                    retry_answer = (await self._llm_service.generate(retry_prompt)).strip()
                    if retry_answer:
                        answer = retry_answer
                    if self._looks_suspicious_calculation_answer(
                        question=question_for_reasoning,
                        context=context,
                        answer=answer,
                    ):
                        answer = NO_CONTEXT_ANSWER
        else:
            base_prompt = build_rag_prompt(
                question=question,
                context=context,
                conversation_history=history_text,
            )
            draft_answer = (await self._llm_service.generate(base_prompt)).strip()
            if not draft_answer or draft_answer == NO_CONTEXT_ANSWER:
                answer = draft_answer or NO_CONTEXT_ANSWER
            else:
                refine_prompt = build_refine_rag_prompt(
                    question=question,
                    context=context,
                    draft_answer=draft_answer,
                    conversation_history=history_text,
                )
                refined_answer = (await self._llm_service.generate(refine_prompt)).strip()
                answer = refined_answer or draft_answer

        answer = self._normalize_answer_text(answer)
        return answer or NO_CONTEXT_ANSWER

    async def answer(
        self,
        *,
        question: str,
        document_ids: list[UUID] | None,
        top_k: int,
        use_hybrid_search: bool,
        use_reranker: bool,
        history: list[ChatTurn] | None = None,
    ) -> ChatResponse:
        history_text = self._build_history_text(history)
        cache_key = self._build_cache_key(
            question,
            document_ids,
            top_k,
            use_hybrid_search,
            use_reranker,
            history_text,
        )
        cached = await self._cache_service.get_json(cache_key)
        if cached:
            cached_payload = dict(cached)
            cached_payload["cached"] = True
            return ChatResponse(**cached_payload)

        question_for_reasoning = f"{history_text}\n{question}".strip() if history_text else question
        missing_scores_message = self._build_missing_nmt_scores_message(question_for_reasoning)
        missing_formula_message = self._build_missing_nmt_formula_message(question_for_reasoning)
        effective_top_k = max(top_k, 6) if self._is_calculation_question(question_for_reasoning) else top_k
        retrieval_query = self._build_retrieval_query(question, history_text)
        retrieved = await self._retrieval_service.retrieve(
            retrieval_query,
            top_k=effective_top_k,
            document_ids=document_ids,
            use_hybrid_search=use_hybrid_search,
            use_reranker=use_reranker,
        )

        if not retrieved:
            deterministic_without_context = self._build_deterministic_nmt_answer(
                question=question_for_reasoning,
                context=question_for_reasoning,
            )
            if deterministic_without_context:
                response = ChatResponse(
                    answer=deterministic_without_context,
                    sources=[],
                    used_top_k=effective_top_k,
                )
                await self._cache_service.set_json(
                    cache_key,
                    response.model_dump(mode="json", exclude={"cached"}),
                )
                return response

            response = ChatResponse(
                answer=missing_scores_message or missing_formula_message or NO_CONTEXT_ANSWER,
                sources=[],
                used_top_k=effective_top_k,
            )
            await self._cache_service.set_json(
                cache_key,
                response.model_dump(mode="json", exclude={"cached"}),
            )
            return response

        context = self._build_context(retrieved)
        answer = await self._generate_answer_text(
            question=question,
            context=context,
            history_text=history_text,
        )

        response = ChatResponse(
            answer=answer,
            sources=self._build_sources(retrieved),
            used_top_k=effective_top_k,
        )
        await self._cache_service.set_json(
            cache_key,
            response.model_dump(mode="json", exclude={"cached"}),
        )
        return response

    async def stream_answer(
        self,
        *,
        question: str,
        document_ids: list[UUID] | None,
        top_k: int,
        use_hybrid_search: bool,
        use_reranker: bool,
        history: list[ChatTurn] | None = None,
    ) -> AsyncGenerator[dict, None]:
        history_text = self._build_history_text(history)
        question_for_reasoning = f"{history_text}\n{question}".strip() if history_text else question
        missing_scores_message = self._build_missing_nmt_scores_message(question_for_reasoning)
        missing_formula_message = self._build_missing_nmt_formula_message(question_for_reasoning)
        effective_top_k = max(top_k, 6) if self._is_calculation_question(question_for_reasoning) else top_k
        retrieval_query = self._build_retrieval_query(question, history_text)
        retrieved = await self._retrieval_service.retrieve(
            retrieval_query,
            top_k=effective_top_k,
            document_ids=document_ids,
            use_hybrid_search=use_hybrid_search,
            use_reranker=use_reranker,
        )

        if not retrieved:
            deterministic_without_context = self._build_deterministic_nmt_answer(
                question=question_for_reasoning,
                context=question_for_reasoning,
            )
            answer_text = (
                deterministic_without_context
                or missing_scores_message
                or missing_formula_message
                or NO_CONTEXT_ANSWER
            )
            yield {"event": "chunk", "data": {"token": answer_text}}
            yield {"event": "sources", "data": {"sources": []}}
            yield {"event": "done", "data": {"used_top_k": effective_top_k}}
            return

        context = self._build_context(retrieved)
        answer_text = await self._generate_answer_text(
            question=question,
            context=context,
            history_text=history_text,
        )
        for token in answer_text.split():
            yield {"event": "chunk", "data": {"token": f"{token} "}}

        yield {
            "event": "sources",
            "data": {"sources": [source.model_dump(mode="json") for source in self._build_sources(retrieved)]},
        }
        yield {"event": "done", "data": {"used_top_k": effective_top_k}}
