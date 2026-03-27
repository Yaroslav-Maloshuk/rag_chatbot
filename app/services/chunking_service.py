from __future__ import annotations

import logging

from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer

from app.core.config import get_settings
from app.services.types import ChunkPayload, ExtractedPage

logger = logging.getLogger(__name__)
settings = get_settings()


class ChunkingService:
    def __init__(self) -> None:
        self._tokenizer = None
        self._splitter = self._build_splitter()

    def _build_splitter(self) -> RecursiveCharacterTextSplitter:
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(settings.chunk_tokenizer_name)
            return RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
                tokenizer=self._tokenizer,
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
                separators=["\n\n", "\n", ". ", " ", ""],
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Falling back to character chunking because tokenizer load failed: %s", exc
            )
            return RecursiveCharacterTextSplitter(
                chunk_size=settings.chunk_size * 4,
                chunk_overlap=settings.chunk_overlap * 4,
                separators=["\n\n", "\n", ". ", " ", ""],
            )

    def _count_tokens(self, text: str) -> int:
        if self._tokenizer is None:
            return len(text.split())
        return len(self._tokenizer.encode(text, add_special_tokens=False))

    def chunk_pages(self, pages: list[ExtractedPage]) -> list[ChunkPayload]:
        if not pages:
            return []

        chunks: list[ChunkPayload] = []
        chunk_index = 0

        for page in pages:
            if not page.text.strip():
                continue

            page_chunks = self._splitter.split_text(page.text)
            for chunk_text in page_chunks:
                clean_chunk = chunk_text.strip()
                if not clean_chunk:
                    continue
                chunks.append(
                    ChunkPayload(
                        page_number=page.page_number,
                        chunk_index=chunk_index,
                        text=clean_chunk,
                        token_count=self._count_tokens(clean_chunk),
                    )
                )
                chunk_index += 1

        return chunks
