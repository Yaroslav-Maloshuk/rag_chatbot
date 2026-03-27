from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)


class ChatRequest(BaseModel):
    question: str = Field(min_length=3, max_length=4000)
    document_ids: list[UUID] | None = None
    top_k: int | None = Field(default=None, ge=1, le=20)
    use_hybrid_search: bool | None = None
    use_reranker: bool | None = None
    history: list[ChatTurn] | None = Field(default=None, max_length=24)


class SourceChunk(BaseModel):
    document_id: UUID
    filename: str
    page_number: int
    chunk_index: int
    score: float
    text_preview: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceChunk] = Field(default_factory=list)
    used_top_k: int
    cached: bool = False


class StreamChunk(BaseModel):
    token: str
