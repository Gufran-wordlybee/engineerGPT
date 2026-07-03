"""
preprocessing.split_sections
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Splits a book PDF into section-level JSON files based on extracted TOC entries.

Each section is saved as a standalone JSON document under
``<output_dir>/sections/<section_id>.json`` and contains the full text
extracted from its page range, along with placeholder lists for images and
equations that downstream modules populate.

Public API
----------
split_sections(pdf_path, toc, output_dir) -> list[dict]
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pymupdf  # PyMuPDF


# ─── helpers ────────────────────────────────────────────────────────────────


def _slugify(title: str, max_length: int = 60) -> str:
    """Convert a human-readable title into a filesystem-safe slug.

    Rules:
        1. Lowercase everything.
        2. Replace whitespace / dots with hyphens.
        3. Strip all characters that are not alphanumeric or hyphens.
        4. Collapse consecutive hyphens and trim leading/trailing hyphens.
        5. Truncate to *max_length* characters.

    Examples
    --------
    >>> _slugify("3.2 Fluid Dynamics")
    '3-2-fluid-dynamics'
    >>> _slugify("Chapter 10: Boundary—Layer Theory!!!")
    'chapter-10-boundary-layer-theory'
    """
    slug = title.lower()
    slug = re.sub(r"[\s.]+", "-", slug)           # spaces & dots → hyphens
    slug = re.sub(r"[^a-z0-9-]", "", slug)         # drop everything else
    slug = re.sub(r"-{2,}", "-", slug)              # collapse repeated hyphens
    slug = slug.strip("-")
    return slug[:max_length]


def _find_parent_chapter(toc: list[dict], current_index: int) -> str:
    """Return the title of the nearest preceding TOC entry with ``level == 1``.

    If no level-1 entry precedes *current_index* the current entry's own
    title is returned as a fallback.
    """
    for i in range(current_index - 1, -1, -1):
        if toc[i]["level"] == 1:
            return toc[i]["title"]
    # Fallback: if the current entry itself is level 1, use it; otherwise
    # return a sensible default.
    if toc[current_index]["level"] == 1:
        return toc[current_index]["title"]
    return "Untitled Chapter"


# ─── public API ─────────────────────────────────────────────────────────────


def split_sections(
    pdf_path: str,
    toc: list[dict],
    output_dir: str,
) -> list[dict]:
    """Split a book into per-section JSON files.

    Parameters
    ----------
    pdf_path : str
        Filesystem path to the source PDF.
    toc : list[dict]
        Table-of-contents entries produced by ``extract_toc``.  Each dict
        has keys ``level`` (int), ``title`` (str), and ``start_page`` (int,
        **0-indexed**).
    output_dir : str
        Base output directory for this book (e.g.
        ``books/processed/fluid_mechanics/``).

    Returns
    -------
    list[dict]
        Lightweight section-metadata dicts (no full text) suitable for
        passing to downstream pipeline stages such as ``extract_images``.
    """
    pdf_path_obj = Path(pdf_path)
    output_path = Path(output_dir)
    sections_dir = output_path / "sections"
    sections_dir.mkdir(parents=True, exist_ok=True)

    doc: pymupdf.Document = pymupdf.open(str(pdf_path_obj))
    last_page: int = len(doc) - 1  # 0-indexed

    # Edge case: empty TOC → treat the whole book as one section.
    if not toc:
        toc = [
            {
                "level": 1,
                "title": pdf_path_obj.stem,
                "start_page": 0,
            }
        ]

    metadata_list: list[dict] = []

    for idx, entry in enumerate(toc):
        start_page: int = entry["start_page"]

        # Determine end_page: next entry's start_page - 1, or last page.
        if idx + 1 < len(toc):
            end_page: int = toc[idx + 1]["start_page"] - 1
        else:
            end_page = last_page

        # Clamp to valid range.
        start_page = max(0, min(start_page, last_page))
        end_page = max(start_page, min(end_page, last_page))

        section_id: str = _slugify(entry["title"])
        chapter: str = _find_parent_chapter(toc, idx)

        print(
            f"[SPLIT] Processing section: {entry['title']} "
            f"(pages {start_page}-{end_page})"
        )

        # ── extract text ────────────────────────────────────────────────
        text_parts: list[str] = []
        for page_num in range(start_page, end_page + 1):
            page: pymupdf.Page = doc[page_num]
            text_parts.append(page.get_text("text"))
        full_text: str = "\n".join(text_parts)

        # ── build section document ──────────────────────────────────────
        section_doc: dict = {
            "section_id": section_id,
            "title": entry["title"],
            "level": entry["level"],
            "chapter": chapter,
            "start_page": start_page,
            "end_page": end_page,
            "text": full_text,
            "images": [],
            "equations": [],
        }

        # ── persist to JSON ─────────────────────────────────────────────
        json_path: Path = sections_dir / f"{section_id}.json"
        json_path.write_text(
            json.dumps(section_doc, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        # ── lightweight metadata for return value ───────────────────────
        metadata_list.append(
            {
                "section_id": section_id,
                "title": entry["title"],
                "level": entry["level"],
                "chapter": chapter,
                "start_page": start_page,
                "end_page": end_page,
            }
        )

    doc.close()
    return metadata_list


# ─── standalone testing ─────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print(
            "Usage: python -m preprocessing.split_sections "
            "<pdf_path> <output_dir> [toc_json_path]"
        )
        print(
            "\nIf toc_json_path is omitted the entire book is treated as a "
            "single section."
        )
        sys.exit(1)

    _pdf = sys.argv[1]
    _out = sys.argv[2]

    _toc: list[dict] = []
    if len(sys.argv) >= 4:
        _toc = json.loads(Path(sys.argv[3]).read_text(encoding="utf-8"))

    result = split_sections(_pdf, _toc, _out)
    print(f"\n✓ Created {len(result)} section(s).")
    for sec in result:
        print(f"  • [{sec['section_id']}] {sec['title']} "
              f"(pp. {sec['start_page']}–{sec['end_page']})")
