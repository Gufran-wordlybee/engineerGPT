"""
preprocessing.extract_images
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Extracts **three** kinds of visual content from each section's pages:

1. **Embedded raster images** — JPEG/PNG already embedded in the PDF.
2. **Vector diagrams** — drawn with PDF paths/rects/arrows (e.g. TikZ,
   PowerPoint exports).  Detected via ``page.get_drawings()``, clustered
   by spatial proximity, then rasterized with ``page.get_pixmap(clip=…)``.
3. **Display equations** — numbered (``(3.14)`` at end-of-line) or
   unnumbered (math-font/math-symbol dense lines).  Rasterized the same
   way.  Inline single-symbol equations are deliberately skipped.

Updates each section JSON's ``images`` and ``equations`` fields with
structured metadata dicts.

Public API
----------
extract_images(pdf_path, output_dir, sections) -> int
"""

from __future__ import annotations

import json
import re
import warnings
from pathlib import Path
from typing import Any

import pymupdf  # PyMuPDF

from config.settings import (
    MIN_IMAGE_SIZE,
    RASTERIZE_DPI,
    DIAGRAM_MIN_PATHS,
    DIAGRAM_CLUSTER_GAP,
    DIAGRAM_MIN_AREA,
    EQUATION_NUMBER_RE,
    FIGURE_CAPTION_RE,
    MATH_SYMBOL_CHARS,
    MATH_FONT_FRAGMENTS,
)


# ═══════════════════════════════════════════════════════════════════════════
# Pass 1 helpers — embedded raster images
# ═══════════════════════════════════════════════════════════════════════════

def _extract_raster_images(
    doc: pymupdf.Document,
    page: pymupdf.Page,
    seen_xrefs: set[int],
    section_id: str,
    images_dir: Path,
    output_path: Path,
    counter_start: int,
) -> tuple[list[dict[str, Any]], int]:
    """Extract embedded raster images from *page*.

    Returns (list_of_image_metadata_dicts, next_counter_value).
    """
    results: list[dict[str, Any]] = []
    counter = counter_start

    for img_info in page.get_images(full=True):
        xref: int = img_info[0]
        if xref in seen_xrefs:
            continue
        seen_xrefs.add(xref)

        try:
            base_image = doc.extract_image(xref)
        except Exception as exc:
            warnings.warn(
                f"[VISUALS] Could not extract xref {xref} – {exc}",
                stacklevel=2,
            )
            continue

        if base_image is None:
            continue

        width = base_image.get("width", 0)
        height = base_image.get("height", 0)
        if width < MIN_IMAGE_SIZE or height < MIN_IMAGE_SIZE:
            continue

        ext = base_image.get("ext", "png")
        counter += 1
        filename = f"{section_id}_img_{counter}.{ext}"
        dest = images_dir / filename
        dest.write_bytes(base_image["image"])

        results.append({
            "path": str(dest.relative_to(output_path)),
            "type": "raster",
            "caption": None,
        })

    return results, counter


# ═══════════════════════════════════════════════════════════════════════════
# Pass 2 helpers — vector diagrams
# ═══════════════════════════════════════════════════════════════════════════

def _cluster_drawings(
    drawings: list[dict],
    gap: float,
) -> list[tuple[pymupdf.Rect, int]]:
    """Cluster nearby drawing paths into diagram regions.

    Returns a list of (merged_rect, path_count) tuples.
    """
    if not drawings:
        return []

    # Collect valid rects
    rects: list[pymupdf.Rect] = []
    for d in drawings:
        r = d.get("rect")
        if r is None:
            continue
        r = pymupdf.Rect(r)
        if r.is_empty or r.is_infinite:
            continue
        rects.append(r)

    if not rects:
        return []

    # Track which original rects belong to each cluster
    # Start: each rect is its own cluster
    clusters: list[list[int]] = [[i] for i in range(len(rects))]
    merged_rects: list[pymupdf.Rect] = [pymupdf.Rect(r) for r in rects]

    changed = True
    while changed:
        changed = False
        new_clusters: list[list[int]] = []
        new_merged: list[pymupdf.Rect] = []
        used: set[int] = set()

        for i in range(len(merged_rects)):
            if i in used:
                continue
            current_rect = pymupdf.Rect(merged_rects[i])
            current_members = list(clusters[i])

            for j in range(i + 1, len(merged_rects)):
                if j in used:
                    continue
                expanded = pymupdf.Rect(
                    current_rect.x0 - gap,
                    current_rect.y0 - gap,
                    current_rect.x1 + gap,
                    current_rect.y1 + gap,
                )
                candidate = pymupdf.Rect(merged_rects[j])
                if expanded.intersects(candidate):
                    current_rect |= candidate   # union
                    current_members.extend(clusters[j])
                    used.add(j)
                    changed = True

            new_clusters.append(current_members)
            new_merged.append(current_rect)
            used.add(i)

        clusters = new_clusters
        merged_rects = new_merged

    return [
        (merged_rects[i], len(clusters[i]))
        for i in range(len(merged_rects))
    ]


def _find_caption_for_region(
    page: pymupdf.Page,
    region_rect: pymupdf.Rect,
) -> str | None:
    """Search page text for a figure caption near *region_rect*."""
    blocks = page.get_text("dict")["blocks"]
    for block in blocks:
        if block.get("type") != 0:
            continue
        block_rect = pymupdf.Rect(block["bbox"])
        # Caption typically just below or above the figure
        vertical_dist = min(
            abs(block_rect.y0 - region_rect.y1),
            abs(region_rect.y0 - block_rect.y1),
        )
        if vertical_dist > 50:
            continue

        for line in block["lines"]:
            line_text = "".join(s["text"] for s in line["spans"])
            m = FIGURE_CAPTION_RE.search(line_text)
            if m:
                return m.group(0)   # e.g. "Figure 2.1"

    return None


def _extract_vector_diagrams(
    page: pymupdf.Page,
    section_id: str,
    images_dir: Path,
    output_path: Path,
    counter_start: int,
) -> tuple[list[dict[str, Any]], int]:
    """Detect vector diagram regions and rasterize them.

    Returns (list_of_diagram_metadata_dicts, next_counter_value).
    """
    results: list[dict[str, Any]] = []
    counter = counter_start

    try:
        drawings = page.get_drawings()
    except Exception:
        return results, counter

    if not drawings:
        return results, counter

    clusters = _cluster_drawings(drawings, DIAGRAM_CLUSTER_GAP)
    page_rect = page.rect

    for region_rect, path_count in clusters:
        # Filter: enough paths and enough area
        if path_count < DIAGRAM_MIN_PATHS:
            continue
        area = abs(region_rect.width * region_rect.height)
        if area < DIAGRAM_MIN_AREA:
            continue

        # Clip to page bounds
        clip = region_rect & page_rect
        if clip.is_empty:
            continue

        # Look for a caption
        caption = _find_caption_for_region(page, region_rect)

        # Rasterize
        try:
            pix = page.get_pixmap(clip=clip, dpi=RASTERIZE_DPI)
        except Exception as exc:
            warnings.warn(f"[VISUALS] Rasterize failed: {exc}", stacklevel=2)
            continue

        counter += 1
        if caption:
            # Extract the number from caption for filename
            cap_num = FIGURE_CAPTION_RE.search(caption)
            label = cap_num.group(1).replace(".", "-") if cap_num else str(counter)
            filename = f"{section_id}_fig_{label}.png"
        else:
            filename = f"{section_id}_diagram_{counter}.png"

        dest = images_dir / filename
        pix.save(str(dest))

        results.append({
            "path": str(dest.relative_to(output_path)),
            "type": "diagram",
            "caption": caption,
        })

    return results, counter


# ═══════════════════════════════════════════════════════════════════════════
# Pass 3 helpers — display equations
# ═══════════════════════════════════════════════════════════════════════════

_MATH_FONT_RE = re.compile(
    "|".join(re.escape(f) for f in MATH_FONT_FRAGMENTS),
    re.IGNORECASE,
)


def _is_math_line(spans: list[dict]) -> bool:
    """Check if a line is predominantly math content.

    Returns True if the line's dominant font is a math font OR
    if > 15% of non-whitespace chars are math symbols.
    """
    full_text = "".join(s["text"] for s in spans)
    stripped = full_text.replace(" ", "")
    if len(stripped) < 3:
        return False

    # Check font names
    total_chars = 0
    math_font_chars = 0
    for span in spans:
        span_len = len(span["text"].strip())
        total_chars += span_len
        font_name = span.get("font", "")
        if _MATH_FONT_RE.search(font_name):
            math_font_chars += span_len

    if total_chars > 0 and math_font_chars / total_chars > 0.5:
        return True

    # Check math symbol density
    math_chars = sum(1 for c in stripped if c in MATH_SYMBOL_CHARS)
    if len(stripped) > 0 and math_chars / len(stripped) > 0.15:
        return True

    return False


def _detect_equation_regions(
    page: pymupdf.Page,
    page_dict: dict,
) -> list[dict[str, Any]]:
    """Find display equation regions on a page.

    Returns list of {"rect": pymupdf.Rect, "label": str|None}.
    """
    results: list[dict[str, Any]] = []

    for block in page_dict.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block["lines"]:
            spans = line["spans"]
            line_text = "".join(s["text"] for s in spans)
            line_rect = pymupdf.Rect(line["bbox"])

            label: str | None = None

            # Signal 1: numbered equation — line ends with (N.N)
            eq_match = EQUATION_NUMBER_RE.search(line_text.strip())
            if eq_match:
                label = eq_match.group(1)
                results.append({"rect": line_rect, "label": label})
                continue

            # Signal 2 & 3: math-font or math-symbol dense line
            if _is_math_line(spans):
                results.append({"rect": line_rect, "label": None})

    return results


def _extract_equations(
    page: pymupdf.Page,
    section_id: str,
    images_dir: Path,
    output_path: Path,
    counter_start: int,
) -> tuple[list[dict[str, Any]], int]:
    """Detect and rasterize display equations.

    Returns (list_of_equation_metadata_dicts, next_counter_value).
    """
    results: list[dict[str, Any]] = []
    counter = counter_start

    try:
        page_dict = page.get_text("dict")
    except Exception:
        return results, counter

    eq_regions = _detect_equation_regions(page, page_dict)
    page_rect = page.rect

    for region in eq_regions:
        rect: pymupdf.Rect = pymupdf.Rect(region["rect"])
        label = region["label"]

        # Add padding
        padded = pymupdf.Rect(
            rect.x0 - 10,
            rect.y0 - 10,
            rect.x1 + 10,
            rect.y1 + 10,
        )
        clip = padded & page_rect
        if clip.is_empty:
            continue

        try:
            pix = page.get_pixmap(clip=clip, dpi=RASTERIZE_DPI)
        except Exception:
            continue

        counter += 1
        if label:
            safe_label = label.replace(".", "-")
            filename = f"{section_id}_eq_{safe_label}.png"
        else:
            filename = f"{section_id}_eq_{counter}.png"

        dest = images_dir / filename
        pix.save(str(dest))

        results.append({
            "path": str(dest.relative_to(output_path)),
            "type": "equation",
            "label": label,
        })

    return results, counter


# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════

def extract_images(
    pdf_path: str,
    output_dir: str,
    sections: list[dict],
) -> int:
    """Extract raster images, vector diagrams, and equations from all sections.

    Updates each section's JSON with structured ``images`` and ``equations``
    metadata.  Returns the total number of visual elements extracted.
    """
    output_path = Path(output_dir)
    images_dir = output_path / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    sections_dir = output_path / "sections"

    doc: pymupdf.Document = pymupdf.open(str(pdf_path))
    total_visuals: int = 0
    seen_xrefs: set[int] = set()

    for section in sections:
        section_id: str = section["section_id"]
        title: str = section["title"]
        start_page: int = section["start_page"]
        end_page: int = section["end_page"]

        all_images: list[dict[str, Any]] = []
        all_equations: list[dict[str, Any]] = []

        raster_counter = 0
        diagram_counter = 0
        equation_counter = 0

        for page_num in range(start_page, end_page + 1):
            page = doc[page_num]

            # Pass 1: embedded raster images
            raster_results, raster_counter = _extract_raster_images(
                doc, page, seen_xrefs, section_id,
                images_dir, output_path, raster_counter,
            )
            all_images.extend(raster_results)

            # Pass 2: vector diagrams
            diagram_results, diagram_counter = _extract_vector_diagrams(
                page, section_id,
                images_dir, output_path, diagram_counter,
            )
            all_images.extend(diagram_results)

            # Pass 3: display equations
            equation_results, equation_counter = _extract_equations(
                page, section_id,
                images_dir, output_path, equation_counter,
            )
            all_equations.extend(equation_results)

        # Update section JSON
        json_path = sections_dir / f"{section_id}.json"
        if json_path.exists():
            section_doc = json.loads(json_path.read_text(encoding="utf-8"))
            section_doc["images"] = all_images
            section_doc["equations"] = all_equations
            json_path.write_text(
                json.dumps(section_doc, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        n_raster = sum(1 for i in all_images if i["type"] == "raster")
        n_diagram = sum(1 for i in all_images if i["type"] == "diagram")
        n_eq = len(all_equations)
        section_total = n_raster + n_diagram + n_eq
        total_visuals += section_total

        print(
            f"[VISUALS] Section '{title}': "
            f"{n_raster} raster images, {n_diagram} diagrams, {n_eq} equations"
        )

    doc.close()
    return total_visuals


# ═══════════════════════════════════════════════════════════════════════════
# CLI entry-point
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(
        "Usage:\n"
        "  This module runs as part of the preprocessing pipeline.\n\n"
        "  from preprocessing.split_sections import split_sections\n"
        "  from preprocessing.extract_images import extract_images\n\n"
        "  sections = split_sections(pdf_path, toc, output_dir)\n"
        "  total = extract_images(pdf_path, output_dir, sections)\n"
    )
