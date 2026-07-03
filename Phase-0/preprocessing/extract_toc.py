"""
preprocessing.extract_toc
~~~~~~~~~~~~~~~~~~~~~~~~~
Extract the Table of Contents from a PDF using a 3-tier strategy:

    Tier 1 — Embedded bookmarks  (fastest, most reliable when present)
    Tier 2 — Font-size heuristics (works on most professionally typeset PDFs)
    Tier 3 — Regex patterns       (last-resort fallback for plain-text headings)

Public API
----------
    extract_toc(pdf_path) -> list[dict]

Each returned entry has the shape::

    {"level": int, "title": str, "start_page": int}

Pages are 0-indexed (PyMuPDF convention).
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from typing import Any

import pymupdf                       # PyMuPDF >= 1.24

from config.settings import (
    HEADING_FONT_RATIO_L1,
    HEADING_FONT_RATIO_L2,
    HEADING_REGEX_PATTERNS,
    MIN_BOOKMARK_ENTRIES,
    MIN_HEADING_FONT_RATIO,
)

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------
TOCEntry = dict[str, Any]            # {"level": int, "title": str, "start_page": int}


# ═══════════════════════════════════════════════════════════════════════════
# Tier 1 — Bookmarks
# ═══════════════════════════════════════════════════════════════════════════

def _extract_from_bookmarks(doc: pymupdf.Document) -> list[TOCEntry]:
    """Return TOC entries from embedded PDF bookmarks (``doc.get_toc``).

    PyMuPDF returns ``[level, title, page_1based]`` triples.  We convert to
    0-based pages and only accept the result when the bookmark count meets
    the configured minimum.
    """
    raw_toc = doc.get_toc()          # list of [level, title, page_1based]

    if len(raw_toc) < MIN_BOOKMARK_ENTRIES:
        return []

    entries: list[TOCEntry] = []
    for level, title, page_1 in raw_toc:
        # Some bookmarks point to page 0 or negative — clamp to valid range.
        page_0 = max(page_1 - 1, 0)
        entries.append({
            "level": int(level),
            "title": title.strip(),
            "start_page": page_0,
        })

    entries.sort(key=lambda e: e["start_page"])
    return entries


# ═══════════════════════════════════════════════════════════════════════════
# Tier 2 — Font-size heuristics
# ═══════════════════════════════════════════════════════════════════════════

def _body_font_size(doc: pymupdf.Document) -> float:
    """Determine the most common (body-text) font size across all pages.

    Each span's font size is rounded to the nearest 0.5 pt, and the size
    with the highest total character count is chosen as the *body size*.
    """
    histogram: Counter[float] = Counter()

    for page in doc:
        blocks = page.get_text("dict", flags=pymupdf.TEXT_PRESERVE_WHITESPACE)["blocks"]
        for block in blocks:
            if block.get("type") != 0:       # 0 = text block
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    text = span["text"].strip()
                    if not text:
                        continue
                    size_rounded = round(span["size"] * 2) / 2
                    histogram[size_rounded] += len(text)

    if not histogram:
        return 0.0

    # Mode — the size with the largest total character weight.
    return histogram.most_common(1)[0][0]


def _assign_level(font_size: float, body_size: float) -> int:
    """Map a heading font size to a structural level (1–3)."""
    ratio = font_size / body_size
    if ratio >= HEADING_FONT_RATIO_L1:
        return 1
    if ratio >= HEADING_FONT_RATIO_L2:
        return 2
    return 3


_BOLD_FONT_NAMES = re.compile(r"bold|black|heavy", re.IGNORECASE)
_PYMUPDF_BOLD_FLAG = 1 << 4         # bit 4 = bold in span["flags"]


def _extract_from_font_size(doc: pymupdf.Document) -> list[TOCEntry]:
    """Identify headings by comparing each span's font size to the body size.

    Spans whose size is >= ``body_size * MIN_HEADING_FONT_RATIO`` — or that
    are bold with size >= ``body_size * 1.05`` — are treated as heading
    candidates.  Adjacent spans on the same line are merged into a single
    heading string.
    """
    body_size = _body_font_size(doc)
    if body_size <= 0:
        return []

    heading_threshold = body_size * MIN_HEADING_FONT_RATIO
    bold_threshold = body_size * 1.05

    entries: list[TOCEntry] = []

    for page_idx, page in enumerate(doc):
        blocks = page.get_text("dict", flags=pymupdf.TEXT_PRESERVE_WHITESPACE)["blocks"]
        for block in blocks:
            if block.get("type") != 0:
                continue
            for line in block["lines"]:
                # Collect heading spans in this line, tracking the max size
                # and the top-most y coordinate for sorting.
                heading_spans: list[str] = []
                max_size: float = 0.0
                y_pos: float = line["bbox"][1]     # top-y of the line

                for span in line["spans"]:
                    text = span["text"]
                    size = span["size"]
                    flags = span.get("flags", 0)
                    font_name = span.get("font", "")

                    is_large = size >= heading_threshold
                    is_bold = bool(
                        (flags & _PYMUPDF_BOLD_FLAG)
                        or _BOLD_FONT_NAMES.search(font_name)
                    )
                    is_bold_heading = is_bold and size >= bold_threshold

                    if is_large or is_bold_heading:
                        heading_spans.append(text)
                        max_size = max(max_size, size)

                if not heading_spans:
                    continue

                merged = " ".join(heading_spans).strip()
                # Collapse internal whitespace runs.
                merged = re.sub(r"\s+", " ", merged)

                # Skip very short artifacts (page numbers, bullet chars …).
                if len(merged) < 3:
                    continue

                entries.append({
                    "level": _assign_level(max_size, body_size),
                    "title": merged,
                    "start_page": page_idx,
                    "_y": y_pos,               # private; used only for sorting
                })

    # Must find a meaningful number of headings to trust this heuristic.
    if len(entries) < 3:
        return []

    # Sort by page, then vertical position on the page.
    entries.sort(key=lambda e: (e["start_page"], e["_y"]))

    # Strip the internal sort key before returning.
    for entry in entries:
        entry.pop("_y", None)

    return entries


# ═══════════════════════════════════════════════════════════════════════════
# Tier 3 — Regex patterns
# ═══════════════════════════════════════════════════════════════════════════

_NON_NUMERIC = re.compile(r"[^\d\s\.\-:]+")


def _extract_from_regex(doc: pymupdf.Document) -> list[TOCEntry]:
    """Fall back to regex matching on raw page text.

    Patterns are tried in order from most specific (sub-sub-section) to
    least specific (chapter/part) so that e.g. ``3.2.1 …`` isn't captured
    by the ``\\d+\\.\\d+`` pattern first.
    """
    entries: list[TOCEntry] = []

    for page_idx, page in enumerate(doc):
        text = page.get_text("text")
        if not text:
            continue
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue

            # Skip lines that are essentially bare page numbers.
            non_num_chars = _NON_NUMERIC.findall(stripped)
            if sum(len(c) for c in non_num_chars) < 3:
                continue

            for pattern, level in HEADING_REGEX_PATTERNS:
                if pattern.search(stripped):
                    entries.append({
                        "level": level,
                        "title": stripped,
                        "start_page": page_idx,
                    })
                    break                    # first matching pattern wins

    entries.sort(key=lambda e: e["start_page"])
    return entries


# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════

def extract_toc(pdf_path: str) -> list[TOCEntry]:
    """Extract a Table of Contents from *pdf_path* using a 3-tier strategy.

    Parameters
    ----------
    pdf_path:
        Filesystem path to the PDF file.

    Returns
    -------
    list[dict]
        Each dict contains ``level`` (int), ``title`` (str), and
        ``start_page`` (int, 0-indexed).
    """
    doc = pymupdf.open(pdf_path)

    # Guard: empty or single-page PDF with no content.
    if doc.page_count == 0:
        print("[TOC] PDF has no pages — returning empty TOC")
        doc.close()
        return []

    # Tier 1: Bookmarks ─────────────────────────────────────────────────
    toc = _extract_from_bookmarks(doc)
    if toc:
        print(f"[TOC] Extracted {len(toc)} entries from bookmarks")
        doc.close()
        return toc

    # Tier 2: Font-size heuristics ──────────────────────────────────────
    toc = _extract_from_font_size(doc)
    if toc:
        print(f"[TOC] Extracted {len(toc)} entries from font-size analysis")
        doc.close()
        return toc

    # Tier 3: Regex fallback ────────────────────────────────────────────
    toc = _extract_from_regex(doc)
    print(f"[TOC] Extracted {len(toc)} entries from regex patterns")
    doc.close()
    return toc


# ═══════════════════════════════════════════════════════════════════════════
# CLI entry-point:  python -m preprocessing.extract_toc path/to/book.pdf
# ═══════════════════════════════════════════════════════════════════════════

def _pretty_print_toc(toc: list[TOCEntry]) -> None:
    """Print TOC entries with indentation reflecting heading level."""
    if not toc:
        print("  (no entries)")
        return
    for entry in toc:
        indent = "  " * (entry["level"] - 1)
        page = entry["start_page"]
        print(f"  {indent}[L{entry['level']}]  p.{page:<5}  {entry['title']}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m preprocessing.extract_toc <path/to/book.pdf>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    print(f"[TOC] Processing: {pdf_path}\n")

    result = extract_toc(pdf_path)

    print(f"\n{'─' * 60}")
    print(f"  Table of Contents  ({len(result)} entries)")
    print(f"{'─' * 60}")
    _pretty_print_toc(result)
    print(f"{'─' * 60}")
