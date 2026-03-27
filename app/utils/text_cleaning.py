from __future__ import annotations

import re

MULTI_WHITESPACE_RE = re.compile(r"[ \t]+")
MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")


def clean_text(text: str) -> str:
    if not text:
        return ""

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = CONTROL_CHAR_RE.sub(" ", normalized)
    normalized = MULTI_WHITESPACE_RE.sub(" ", normalized)
    normalized = MULTI_NEWLINE_RE.sub("\n\n", normalized)
    normalized = normalized.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")
    return normalized.strip()
