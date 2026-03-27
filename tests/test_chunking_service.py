import pytest

pytest.importorskip("langchain_text_splitters")
pytest.importorskip("transformers")

from app.services.chunking_service import ChunkingService
from app.services.types import ExtractedPage


def test_chunking_returns_empty_for_no_pages() -> None:
    service = ChunkingService()
    chunks = service.chunk_pages([])
    assert chunks == []


def test_chunking_splits_non_empty_page() -> None:
    service = ChunkingService()
    text = " ".join(["token"] * 3000)
    pages = [ExtractedPage(page_number=1, text=text)]

    chunks = service.chunk_pages(pages)

    assert len(chunks) >= 2
    assert all(chunk.page_number == 1 for chunk in chunks)
    assert chunks[0].chunk_index == 0
