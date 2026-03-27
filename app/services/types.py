from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID


@dataclass(slots=True)
class ExtractedPage:
    page_number: int
    text: str


@dataclass(slots=True)
class ChunkPayload:
    page_number: int
    chunk_index: int
    text: str
    token_count: int


@dataclass(slots=True)
class RetrievalResult:
    document_id: UUID
    filename: str
    page_number: int
    chunk_index: int
    text: str
    score: float
