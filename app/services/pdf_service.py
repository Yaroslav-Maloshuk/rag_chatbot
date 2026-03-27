from __future__ import annotations

import logging
from pathlib import Path
import subprocess

import fitz

from app.core.config import get_settings
from app.services.types import ExtractedPage
from app.utils.text_cleaning import clean_text

logger = logging.getLogger(__name__)
settings = get_settings()


class PDFExtractionError(Exception):
    pass


class PDFService:
    def __init__(self) -> None:
        self._available_ocr_langs: set[str] | None = None

    def _list_tesseract_languages(self) -> set[str]:
        if self._available_ocr_langs is not None:
            return self._available_ocr_langs

        langs: set[str] = set()
        try:
            result = subprocess.run(
                ["tesseract", "--list-langs"],
                check=True,
                capture_output=True,
                text=True,
            )
            lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
            for line in lines:
                if line.startswith("List of available languages"):
                    continue
                langs.add(line)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Unable to list Tesseract languages: %s", exc)

        self._available_ocr_langs = langs
        return langs

    def _resolve_ocr_candidates(self) -> list[str]:
        available = self._list_tesseract_languages()

        raw_candidates: list[str] = []
        if settings.ocr_language:
            raw_candidates.extend(
                [item.strip() for item in settings.ocr_language.split(",") if item.strip()]
            )
        if settings.ocr_language_candidates:
            raw_candidates.extend(
                [
                    item.strip()
                    for item in settings.ocr_language_candidates.split(",")
                    if item.strip()
                ]
            )

        candidates: list[str] = []
        for candidate in raw_candidates:
            if candidate.lower() == "auto":
                # Prioritize script models for broad multilingual OCR.
                script_models = [
                    "script/Latin",
                    "script/Cyrillic",
                    "script/Arabic",
                    "script/Devanagari",
                    "script/HanS",
                    "script/HanT",
                    "script/Japanese",
                    "script/Korean",
                    "script/Thai",
                    "script/Hebrew",
                    "script/Greek",
                ]
                for script_model in script_models:
                    if not available or script_model in available:
                        candidates.append(script_model)
                if not available or "eng" in available:
                    candidates.append("eng")
                continue

            if "+" in candidate:
                parts = [part.strip() for part in candidate.split("+") if part.strip()]
                if not parts:
                    continue
                if not available or all(part in available for part in parts):
                    candidates.append("+".join(parts))
                continue

            if not available or candidate in available:
                candidates.append(candidate)

        # Deterministic de-duplication preserving order.
        deduped: list[str] = []
        seen: set[str] = set()
        for candidate in candidates:
            if candidate in seen:
                continue
            seen.add(candidate)
            deduped.append(candidate)

        return deduped or ["eng"]

    def _extract_block_text(self, page: fitz.Page) -> str:
        blocks = page.get_text("blocks")
        merged = "\n".join(block[4] for block in blocks if len(block) > 4 and str(block[4]).strip())
        return clean_text(merged)

    def _extract_ocr_text(self, page: fitz.Page, page_number: int) -> str:
        if not settings.enable_ocr_fallback:
            return ""

        best_text = ""
        best_lang = ""

        for language in self._resolve_ocr_candidates():
            try:
                text_page = page.get_textpage_ocr(
                    language=language,
                    dpi=settings.ocr_dpi,
                    full=True,
                )
                text = clean_text(page.get_text("text", textpage=text_page))
                if len(text) > len(best_text):
                    best_text = text
                    best_lang = language
                if len(best_text) >= settings.ocr_good_text_min_chars:
                    break
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "OCR extraction failed for page %s with language %s: %s",
                    page_number,
                    language,
                    exc,
                )

        if best_text:
            logger.info("OCR extracted page %s using language model: %s", page_number, best_lang)
        return best_text

    def extract_pages(self, file_path: str | Path) -> tuple[list[ExtractedPage], int]:
        path = Path(file_path)
        if not path.exists():
            raise PDFExtractionError(f"PDF file not found: {path}")

        pages: list[ExtractedPage] = []
        total_pages = 0

        try:
            with fitz.open(path) as pdf:
                total_pages = pdf.page_count
                for idx, page in enumerate(pdf, start=1):
                    raw_text = page.get_text("text")
                    text = clean_text(raw_text)
                    if not text:
                        text = self._extract_block_text(page)
                    if not text:
                        text = self._extract_ocr_text(page, idx)
                    if text:
                        pages.append(ExtractedPage(page_number=idx, text=text))
        except Exception as exc:  # noqa: BLE001
            logger.exception("PDF extraction failed for %s", path)
            raise PDFExtractionError(str(exc)) from exc

        return pages, total_pages
