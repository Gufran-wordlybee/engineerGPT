"""
preprocessing.split_sections
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Splits a book PDF into section-level JSON files based on extracted TOC entries.

Now includes:
- **Hierarchical structure**: each section stores parent_id and children
- **Header/footer stripping**: repeating page headers/footers are detected and removed
- **Auto-merge tiny sections**: sections below MIN_SECTION_CHARS merged into siblings
- **Length QA report**: flags suspiciously long/short sections for manual review

Public API
----------
split_sections(pdf_path, toc, output_dir) -> list[dict]
"""

from __future__ import annotations

import json
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

import pymupdf  # PyMuPDF

from config.settings import (
    MIN_SECTION_CHARS,
    HEADER_FOOTER_LINES,
    HEADER_FOOTER_MIN_RATIO,
    SECTION_OUTLIER_LONG,
    SECTION_OUTLIER_SHORT,
)


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _slugify(title: str, max_length: int = 60) -> str:
    """Convert a human-readable title into a filesystem-safe slug.

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
    """Return the title of the nearest preceding TOC entry with ``level == 1``."""
    for i in range(current_index - 1, -1, -1):
        if toc[i]["level"] == 1:
            return toc[i]["title"]
    if toc[current_index]["level"] == 1:
        return toc[current_index]["title"]
    return "Untitled Chapter"


def _find_parent_id(
    toc: list[dict],
    section_ids: list[str],
    current_index: int,
) -> str | None:
    """Find the section_id of the nearest preceding entry with a lower level.

    Level 1 entries have no parent (returns None).
    """
    current_level = toc[current_index]["level"]
    if current_level <= 1:
        return None
    for i in range(current_index - 1, -1, -1):
        if toc[i]["level"] < current_level:
            return section_ids[i]
    return None


# ═══════════════════════════════════════════════════════════════════════════
# Header / footer detection
# ═══════════════════════════════════════════════════════════════════════════

_STRIP_DIGITS_RE = re.compile(r"^\d+\s*|\s*\d+$")
_COLLAPSE_WS_RE = re.compile(r"\s+")


def _normalize_line(line: str) -> str:
    """Normalize a line for comparison: strip, collapse whitespace, remove
    leading/trailing page numbers."""
    line = line.strip()
    line = _STRIP_DIGITS_RE.sub("", line)
    line = _COLLAPSE_WS_RE.sub(" ", line).strip()
    return line.lower()


def _detect_repeating_lines(
    doc: pymupdf.Document,
    start_page: int,
    end_page: int,
) -> set[str]:
    """Find lines that repeat across many pages — these are headers/footers.

    Scans the first/last HEADER_FOOTER_LINES lines of each page, counts how
    often each normalized line appears, and returns those exceeding the
    HEADER_FOOTER_MIN_RATIO threshold.

    Requires at least 5 pages to be meaningful.
    """
    num_pages = end_page - start_page + 1
    if num_pages < 5:
        return set()

    line_counts: Counter[str] = Counter()

    for page_num in range(start_page, end_page + 1):
        page_text = doc[page_num].get_text("text")
        lines = page_text.splitlines()

        # Grab top and bottom lines
        candidate_lines: list[str] = []
        candidate_lines.extend(lines[:HEADER_FOOTER_LINES])
        candidate_lines.extend(lines[-HEADER_FOOTER_LINES:])

        seen_this_page: set[str] = set()
        for raw_line in candidate_lines:
            normed = _normalize_line(raw_line)
            if normed and len(normed) >= 3 and normed not in seen_this_page:
                seen_this_page.add(normed)
                line_counts[normed] += 1

    threshold = num_pages * HEADER_FOOTER_MIN_RATIO
    return {line for line, count in line_counts.items() if count >= threshold}


def _strip_headers_footers(page_text: str, repeating: set[str]) -> str:
    """Remove lines from page_text whose normalized form is in *repeating*."""
    if not repeating:
        return page_text
    cleaned: list[str] = []
    for line in page_text.splitlines():
        if _normalize_line(line) not in repeating:
            cleaned.append(line)
    return "\n".join(cleaned)


# ═══════════════════════════════════════════════════════════════════════════
# QA report
# ═══════════════════════════════════════════════════════════════════════════

def _write_qa_report(
    sections: list[dict[str, Any]],
    output_dir: Path,
) -> tuple[int, int]:
    """Write a length QA report and return (too_long_count, too_short_count)."""
    if not sections:
        return 0, 0

    lengths = [len(s.get("text", "")) for s in sections]
    median_len = statistics.median(lengths) if lengths else 0

    too_long: list[dict] = []
    too_short: list[dict] = []

    for sec in sections:
        char_count = len(sec.get("text", ""))
        if median_len > 0:
            ratio = char_count / median_len
        else:
            ratio = 0
        if ratio > SECTION_OUTLIER_LONG:
            too_long.append({**sec, "_ratio": ratio, "_chars": char_count})
        elif ratio < SECTION_OUTLIER_SHORT and median_len > 0:
            too_short.append({**sec, "_ratio": ratio, "_chars": char_count})

    report_lines = [
        "Section Length QA Report",
        "=" * 40,
        f"Median section length: {int(median_len)} chars",
        f"Total sections: {len(sections)}",
        "",
    ]

    if too_long:
        report_lines.append(f"TOO LONG (> {SECTION_OUTLIER_LONG}x median):")
        for s in too_long:
            report_lines.append(
                f"  [{s['section_id']}] {s['_chars']} chars "
                f"({s['_ratio']:.1f}x median) — pages {s['start_page']}-{s['end_page']}"
            )
        report_lines.append("")

    if too_short:
        report_lines.append(f"TOO SHORT (< {SECTION_OUTLIER_SHORT}x median):")
        for s in too_short:
            report_lines.append(
                f"  [{s['section_id']}] {s['_chars']} chars "
                f"({s['_ratio']:.2f}x median) — pages {s['start_page']}-{s['end_page']}"
            )
        report_lines.append("")

    if not too_long and not too_short:
        report_lines.append("No outliers detected.")

    report_path = output_dir / "qa_report.txt"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    return len(too_long), len(too_short)


# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════

def split_sections(
    pdf_path: str,
    toc: list[dict],
    output_dir: str,
) -> list[dict]:
    """Split a book into per-section JSON files with hierarchy and cleanup.

    Parameters
    ----------
    pdf_path : str
        Filesystem path to the source PDF.
    toc : list[dict]
        TOC entries from ``extract_toc``.  Each dict has keys
        ``level``, ``title``, ``start_page`` (0-indexed).
    output_dir : str
        Base output directory for this book.

    Returns
    -------
    list[dict]
        Lightweight section-metadata dicts suitable for downstream stages.
    """
    pdf_path_obj = Path(pdf_path)
    output_path = Path(output_dir)
    sections_dir = output_path / "sections"
    sections_dir.mkdir(parents=True, exist_ok=True)

    doc: pymupdf.Document = pymupdf.open(str(pdf_path_obj))
    last_page: int = len(doc) - 1

    # Edge case: empty TOC → whole book = one section.
    if not toc:
        toc = [{"level": 1, "title": pdf_path_obj.stem, "start_page": 0}]

    # ── Step 0: detect repeating headers/footers (once for entire book) ──
    repeating_lines = _detect_repeating_lines(doc, 0, last_page)
    if repeating_lines:
        print(f"[SPLIT] Detected {len(repeating_lines)} repeating header/footer patterns")

    # ── Step 1: pre-compute section_ids and page ranges ──────────────────
    section_ids: list[str] = []
    seen_slugs: dict[str, int] = {}          # handle duplicate slugs
    for entry in toc:
        slug = _slugify(entry["title"])
        if slug in seen_slugs:
            seen_slugs[slug] += 1
            slug = f"{slug}-{seen_slugs[slug]}"
        else:
            seen_slugs[slug] = 0
        section_ids.append(slug)

    # ── Step 2: build sections with hierarchy ────────────────────────────
    all_sections: list[dict[str, Any]] = []

    for idx, entry in enumerate(toc):
        start_page = max(0, min(entry["start_page"], last_page))
        end_page = (
            toc[idx + 1]["start_page"] - 1
            if idx + 1 < len(toc)
            else last_page
        )
        end_page = max(start_page, min(end_page, last_page))

        section_id = section_ids[idx]
        chapter = _find_parent_chapter(toc, idx)
        parent_id = _find_parent_id(toc, section_ids, idx)

        # Extract and clean text
        text_parts: list[str] = []
        for pn in range(start_page, end_page + 1):
            raw_text = doc[pn].get_text("text")
            cleaned = _strip_headers_footers(raw_text, repeating_lines)
            text_parts.append(cleaned)
        full_text = "\n".join(text_parts)

        section_doc: dict[str, Any] = {
            "section_id": section_id,
            "title": entry["title"],
            "level": entry["level"],
            "chapter": chapter,
            "parent_id": parent_id,
            "children": [],                  # populated in Step 3
            "start_page": start_page,
            "end_page": end_page,
            "text": full_text,
            "images": [],
            "equations": [],
        }
        all_sections.append(section_doc)

        print(
            f"[SPLIT] Processing section: {entry['title']} "
            f"(pages {start_page}-{end_page})"
        )

    doc.close()

    # ── Step 3: populate children lists ──────────────────────────────────
    id_to_section: dict[str, dict] = {s["section_id"]: s for s in all_sections}
    for sec in all_sections:
        pid = sec["parent_id"]
        if pid and pid in id_to_section:
            id_to_section[pid]["children"].append(sec["section_id"])

    # ── Step 4: auto-merge tiny sections ─────────────────────────────────
    merged_ids: set[str] = set()
    for i, sec in enumerate(all_sections):
        if sec["section_id"] in merged_ids:
            continue
        if len(sec["text"]) >= MIN_SECTION_CHARS:
            continue

        # Find next sibling at the same level
        target: dict | None = None
        for j in range(i + 1, len(all_sections)):
            candidate = all_sections[j]
            if candidate["section_id"] in merged_ids:
                continue
            if candidate["level"] == sec["level"]:
                target = candidate
                break
            if candidate["level"] < sec["level"]:
                break   # left the parent scope

        # Fall back to parent
        if target is None and sec["parent_id"] and sec["parent_id"] in id_to_section:
            target = id_to_section[sec["parent_id"]]

        if target is not None and target["section_id"] != sec["section_id"]:
            target["text"] = sec["text"] + "\n" + target["text"]
            merged_ids.add(sec["section_id"])
            # Remove from parent's children list
            pid = sec["parent_id"]
            if pid and pid in id_to_section:
                children = id_to_section[pid]["children"]
                if sec["section_id"] in children:
                    children.remove(sec["section_id"])
            print(
                f"[SPLIT] Merged tiny section '{sec['title']}' "
                f"({len(sec['text'])} chars) into '{target['title']}'"
            )

    # Remove merged sections
    all_sections = [s for s in all_sections if s["section_id"] not in merged_ids]
    id_to_section = {s["section_id"]: s for s in all_sections}

    # ── Step 5: persist JSONs ────────────────────────────────────────────
    for sec in all_sections:
        json_path = sections_dir / f"{sec['section_id']}.json"
        json_path.write_text(
            json.dumps(sec, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # ── Step 6: QA report ────────────────────────────────────────────────
    n_long, n_short = _write_qa_report(all_sections, output_path)
    flagged = n_long + n_short
    if flagged:
        print(
            f"[SPLIT] QA: {flagged} sections flagged "
            f"({n_long} too long, {n_short} too short) — see qa_report.txt"
        )
    else:
        print("[SPLIT] QA: no length outliers detected")

    # ── Return lightweight metadata ──────────────────────────────────────
    return [
        {
            "section_id": s["section_id"],
            "title": s["title"],
            "level": s["level"],
            "chapter": s["chapter"],
            "parent_id": s["parent_id"],
            "children": s["children"],
            "start_page": s["start_page"],
            "end_page": s["end_page"],
        }
        for s in all_sections
    ]


# ═══════════════════════════════════════════════════════════════════════════
# CLI entry-point
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print(
            "Usage: python -m preprocessing.split_sections "
            "<pdf_path> <output_dir> [toc_json_path]"
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
        parent = f" (parent: {sec['parent_id']})" if sec["parent_id"] else ""
        kids = f" [{len(sec['children'])} children]" if sec["children"] else ""
        print(
            f"  • [{sec['section_id']}] {sec['title']} "
            f"(pp. {sec['start_page']}–{sec['end_page']}){parent}{kids}"
        )
