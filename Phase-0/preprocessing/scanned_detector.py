"""
preprocessing.scanned_detector
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Classify a PDF as **scanned** (image-only / OCR-needed) or **text-based**
(already has an extractable text layer suitable for the pymupdf path).

This module is the *fork point* in ``run_pipeline.py``: it decides, for each
incoming book, which of the two preprocessing paths should handle it.

How it works
------------
1. Opens the PDF with pymupdf (already a project dependency).
2. Samples a spread of pages — first 5, middle 5, last 5 — to get a
   representative picture without scanning every page of a 1000-page book.
3. Calls ``page.get_text("text")`` on each sampled page and counts the
   number of characters returned.
4. Computes the **average characters per page** across the sample.
5. Compares against ``SCANNED_DETECTION_MIN_CHARS_PER_PAGE`` from
   ``config/settings.py``.
   - Below threshold → **scanned** (pure-image pages return ~0 chars).
   - At or above threshold → **text** (real text-based pages return hundreds
     to thousands of chars).

Why this is a separate file
---------------------------
The threshold will need tuning as you feed in more varied books (old scans,
partially-OCR'd PDFs, books with very sparse pages, etc.). Keeping
classification logic isolated means threshold adjustments never risk
touching OCR or section-parsing code.

Public API
----------
classify_pdf(pdf_path) -> dict
    Returns ``{"classification": "scanned"|"text", "avg_chars_per_page": float,
    "per_page_chars": [...], "total_pages": int, "pages_sampled": int}``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pymupdf  # PyMuPDF

from config.settings import (
    SCANNED_DETECTION_MIN_CHARS_PER_PAGE,
    FORCE_OCR_MAX_CHARS_PER_PAGE,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _select_sample_pages(total_pages: int, per_group: int = 5) -> list[int]:
    """Pick a representative spread of page indices to sample.

    Strategy: take up to ``per_group`` pages each from the beginning, middle,
    and end of the document.  For short documents (fewer than
    ``3 * per_group`` pages), just sample every page.

    Parameters
    ----------
    total_pages : int
        Total number of pages in the PDF (0-indexed internally, but this
        value is the *count*, e.g. 200 for a 200-page book).
    per_group : int
        How many pages to take from each region (start / middle / end).
        Default is 5, giving up to 15 sampled pages.

    Returns
    -------
    list[int]
        Sorted, deduplicated list of 0-indexed page numbers to sample.
    """
    if total_pages <= per_group * 3:
        # Short book — just sample every page.
        return list(range(total_pages))

    # First `per_group` pages
    start_pages = list(range(per_group))

    # Middle `per_group` pages, centred around the midpoint
    mid = total_pages // 2
    half = per_group // 2
    middle_pages = list(range(mid - half, mid - half + per_group))

    # Last `per_group` pages
    end_pages = list(range(total_pages - per_group, total_pages))

    # Combine, deduplicate, and sort
    all_pages = sorted(set(start_pages + middle_pages + end_pages))

    # Clamp to valid range (defensive — shouldn't be needed, but safe)
    all_pages = [p for p in all_pages if 0 <= p < total_pages]

    return all_pages


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify_pdf(pdf_path: str) -> dict[str, Any]:
    """Classify a PDF as ``"scanned"`` or ``"text"`` based on extractable text.

    Parameters
    ----------
    pdf_path : str
        Filesystem path to the PDF file.

    Returns
    -------
    dict
        A result dict with:
        - ``classification`` — ``"scanned"`` or ``"text"``.
        - ``avg_chars_per_page`` — the average number of extractable
          characters across the sampled pages.
        - ``per_page_chars`` — list of ``(page_number, char_count)`` tuples
          for each sampled page (useful for debugging / threshold tuning).
        - ``total_pages`` — total page count of the PDF.
        - ``pages_sampled`` — how many pages were actually sampled.
        - ``threshold`` — the threshold value used for classification.

    Raises
    ------
    FileNotFoundError
        If *pdf_path* does not exist.
    RuntimeError
        If the PDF cannot be opened.
    """
    pdf_path_obj = Path(pdf_path)
    if not pdf_path_obj.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # Open the PDF
    try:
        doc: pymupdf.Document = pymupdf.open(str(pdf_path_obj))
    except Exception as exc:
        raise RuntimeError(f"Could not open PDF '{pdf_path}': {exc}") from exc

    total_pages: int = len(doc)

    if total_pages == 0:
        doc.close()
        return {
            "classification": "scanned",
            "avg_chars_per_page": 0.0,
            "per_page_chars": [],
            "total_pages": 0,
            "pages_sampled": 0,
            "threshold": SCANNED_DETECTION_MIN_CHARS_PER_PAGE,
        }

    # Pick which pages to sample
    sample_indices = _select_sample_pages(total_pages)

    # Extract text from each sampled page and count characters
    per_page_chars: list[tuple[int, int]] = []
    for page_idx in sample_indices:
        page_text = doc[page_idx].get_text("text")
        # Count non-whitespace characters — whitespace alone shouldn't count
        # as "extractable text" (some scans return a few stray spaces).
        char_count = len(page_text.strip())
        per_page_chars.append((page_idx, char_count))

    doc.close()

    # Compute average characters per sampled page
    total_chars = sum(count for _, count in per_page_chars)
    pages_sampled = len(per_page_chars)
    avg_chars = total_chars / pages_sampled if pages_sampled > 0 else 0.0

    # Classify based on the configured threshold
    #
    # Two-threshold system:
    #   avg < FORCE_OCR_MAX_CHARS_PER_PAGE (default 5)
    #       → "scanned", needs_force_ocr=True
    #       → Pure image scan, no usable text layer at all.
    #       → Marker must re-OCR every page from scratch.
    #
    #   avg >= FORCE_OCR and < SCANNED_DETECTION_MIN_CHARS (default 25)
    #       → "scanned", needs_force_ocr=False
    #       → Has a partial/garbled text layer.
    #       → Marker can try to use it, only re-OCR'ing where needed.
    #       → Significantly faster than full force-OCR.
    #
    #   avg >= SCANNED_DETECTION_MIN_CHARS (default 25)
    #       → "text", needs_force_ocr=False
    #       → Good text layer, use the pymupdf path instead.
    classification = (
        "scanned"
        if avg_chars < SCANNED_DETECTION_MIN_CHARS_PER_PAGE
        else "text"
    )

    # Decide whether Marker should force-OCR every page, or trust
    # whatever text layer already exists and only OCR where needed.
    # This is the single biggest speed lever for scanned books that
    # aren't 100% image-only — skipping force-OCR can cut processing
    # time by 30–70% on books with partial embedded text.
    needs_force_ocr = avg_chars < FORCE_OCR_MAX_CHARS_PER_PAGE

    result: dict[str, Any] = {
        "classification": classification,
        "needs_force_ocr": needs_force_ocr,
        "avg_chars_per_page": round(avg_chars, 1),
        "per_page_chars": per_page_chars,
        "total_pages": total_pages,
        "pages_sampled": pages_sampled,
        "threshold": SCANNED_DETECTION_MIN_CHARS_PER_PAGE,
        "force_ocr_threshold": FORCE_OCR_MAX_CHARS_PER_PAGE,
    }

    # Print a summary for pipeline logs
    force_ocr_label = "force-OCR" if needs_force_ocr else "partial-OCR"
    print(
        f"[DETECTOR] '{Path(pdf_path).name}': "
        f"avg {avg_chars:.0f} chars/page across {pages_sampled} sampled pages "
        f"→ classified as '{classification}' ({force_ocr_label})"
    )

    return result


# ---------------------------------------------------------------------------
# CLI entry-point — useful for standalone testing / threshold tuning
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m preprocessing.scanned_detector <path/to/book.pdf>")
        print()
        print("Classifies a PDF as 'scanned' or 'text' based on extractable")
        print("characters per page.  Use this to test / tune the detection")
        print("threshold before running the full pipeline.")
        sys.exit(1)

    result = classify_pdf(sys.argv[1])

    print(f"\n{'─' * 60}")
    print(f"  Classification Result")
    print(f"{'─' * 60}")
    print(json.dumps(result, indent=2))
    print(f"{'─' * 60}")
