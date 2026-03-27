from pathlib import Path

import pytest

fitz = pytest.importorskip("fitz")

from app.services.pdf_service import PDFService


def test_extract_pages_with_text(tmp_path: Path) -> None:
    pdf_path = tmp_path / "with_text.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Hello from PDF page one")
    doc.save(pdf_path)
    doc.close()

    service = PDFService()
    pages, total_pages = service.extract_pages(pdf_path)

    assert total_pages == 1
    assert len(pages) == 1
    assert pages[0].page_number == 1
    assert "Hello from PDF" in pages[0].text


def test_extract_pages_empty_pdf(tmp_path: Path) -> None:
    pdf_path = tmp_path / "empty.pdf"
    doc = fitz.open()
    doc.new_page()
    doc.save(pdf_path)
    doc.close()

    service = PDFService()
    pages, total_pages = service.extract_pages(pdf_path)

    assert total_pages == 1
    assert pages == []
