"""Clean canonical document text representation rules.

Defines deterministic, auditable text cleaning rules that:
- Clean whitespace and format consistency.
- Repair broken line joins / hyphenated line breaks.
- Remove null bytes, non-printable control characters, and OCR artifacts.
- Detect and suppress repeated running headers/footers across pages.
- Never silently alter facts, reorder elements, or discard semantic content.
"""

from __future__ import annotations

import re
from typing import Any


class TextCleaner:
    """Deterministic, factual-preserving text cleaner for document extraction."""

    # Control chars to drop (preserve \n, \t, \r)
    _CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\ufffd]")

    # Word broken across lines with a trailing hyphen: "impor-\ntant" -> "important"
    # Negative lookbehind and lookahead ensure we don't accidentally join minus signs or bullet hyphens
    _HYPHEN_BREAK_RE = re.compile(r"(?<=[a-zA-Z]{2})-\n\s*(?=[a-zA-Z]{2})")

    # Excessive non-newline whitespace
    _HORIZONTAL_SPACE_RE = re.compile(r"[^\S\r\n]+")

    # Excessive consecutive blank lines (more than 2 newlines)
    _EXCESSIVE_NEWLINES_RE = re.compile(r"\n{3,}")

    @classmethod
    def clean(cls, text: str) -> str:
        """Clean raw extracted text while preserving facts and punctuation."""
        if not text:
            return ""

        # 1. Remove null bytes and non-printable control chars
        cleaned = cls._CONTROL_CHAR_RE.sub("", text)

        # 2. Normalize Windows/Mac line endings to standard Unix \n
        cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")

        # 3. Repair dehyphenation / broken line joins
        cleaned = cls._HYPHEN_BREAK_RE.sub("", cleaned)

        # 4. Normalize horizontal whitespace line-by-line
        lines = [cls._HORIZONTAL_SPACE_RE.sub(" ", line).strip() for line in cleaned.split("\n")]
        cleaned = "\n".join(lines)

        # 5. Collapse excessive consecutive newlines
        cleaned = cls._EXCESSIVE_NEWLINES_RE.sub("\n\n", cleaned)

        return cleaned.strip()

    clean_text = clean

    @classmethod
    def suppress_headers_footers(
        cls, pages: list[dict[str, Any]] | list[str]
    ) -> list[dict[str, Any]] | list[str]:
        """
        Detects and marks/suppresses running headers or footers repeated identically across multiple pages.
        Supports either list[str] or list[dict[str, Any]] with at least {"text": str}.
        """
        if len(pages) < 2:
            return pages

        is_str_list = isinstance(pages[0], str)
        page_dicts: list[dict[str, Any]] = (
            [{"page_number": i + 1, "text": p} for i, p in enumerate(pages)]
            if is_str_list
            else pages  # type: ignore
        )

        # Extract candidates: first 2 lines and last 2 lines of each page
        first_lines: list[str] = []
        last_lines: list[str] = []
        for p in page_dicts:
            lines = [line.strip() for line in p.get("text", "").splitlines() if line.strip()]
            for line_item in lines[:2]:
                first_lines.append(line_item)
            for line_item in lines[-2:]:
                last_lines.append(line_item)

        # Find repeated candidates appearing in >= 50% of pages
        def find_repeated(candidates: list[str]) -> set[str]:
            counts: dict[str, int] = {}
            for c in candidates:
                # Strip dynamic page numbers (e.g. "Page 12", "12", "- 12 -", "Page 1 of 3")
                norm = re.sub(
                    r"\bpage\s*\d+(\s*of\s*\d+)?\b|\b\d+\b", "", c, flags=re.IGNORECASE
                ).strip()
                if len(norm) > 4:  # Meaningful length candidate
                    counts[norm] = counts.get(norm, 0) + 1
            threshold = max(2, len(page_dicts) // 2)
            return {norm for norm, count in counts.items() if count >= threshold}

        repeated_headers = find_repeated(first_lines)
        repeated_footers = find_repeated(last_lines)

        cleaned_pages: list[Any] = []
        for p in page_dicts:
            page_text = p.get("text", "")
            lines = page_text.splitlines()
            new_lines = []
            for i, line in enumerate(lines):
                norm = re.sub(
                    r"\bpage\s*\d+(\s*of\s*\d+)?\b|\b\d+\b", "", line, flags=re.IGNORECASE
                ).strip()
                # Suppress if matching repeated header at top or repeated footer at bottom
                if i < 3 and norm and norm in repeated_headers:
                    continue
                if i >= len(lines) - 3 and norm and norm in repeated_footers:
                    continue
                new_lines.append(line)

            cleaned_text = "\n".join(new_lines).strip()
            if is_str_list:
                cleaned_pages.append(cleaned_text)
            else:
                p_copy = dict(p)
                p_copy["text"] = cleaned_text
                cleaned_pages.append(p_copy)

        return cleaned_pages
