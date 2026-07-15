"""
preprocessing.parse_marker_output
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The **bridge** between Marker's raw JSON output and the existing
``sections/*.json`` schema used by ``build_index.py``, ``core/router.py``,
and ``core/generator.py``.

This module reads Marker's hierarchical block tree, walks it in reading
order, and rewrites it into the exact same per-section JSON format that
``split_sections.py`` produces for text-based PDFs. This is what makes
``build_index.py`` indifferent to which preprocessing path produced a
given book.

Output contract (must match split_sections.py exactly)
------------------------------------------------------
Each section JSON at ``books/processed/<book>/sections/<id>.json``::

    {
        "section_id":  "string, slugified title",
        "title":       "string",
        "level":       int,       # TOC depth (1 = chapter, 2 = section, ...)
        "chapter":     "string",  # parent chapter title
        "parent_id":   "string|null",
        "children":    ["list of child section_ids"],
        "start_page":  int,       # 0-indexed
        "end_page":    int,       # 0-indexed
        "text":        "string",  # full section body text
        "images":      [...],     # image metadata dicts
        "equations":   [...]      # equation metadata dicts
    }

How Marker's JSON maps to sections
-----------------------------------
Marker outputs a tree of typed blocks.  We walk them in document order:

- **SectionHeader** → close the current section accumulator, start a new
  one.  The header text becomes ``title``; the heading level (h1/h2/h3)
  maps to ``level``.
- **Text** / **TextInlineMath** → append to current section's ``text``.
- **Picture** / **Figure** → save the image file, add to ``images`` list.
- **Equation** → save the image file, add to ``equations`` list.  This is
  a huge win over the pymupdf path, which cannot distinguish equations
  from regular figures.
- **Table** → render as text and append to section ``text``.
- **ListItem** / **ListGroup** / **Caption** → append text content.

Page numbers are extracted from block IDs (format: ``/page/<n>/...``)
so ``start_page`` / ``end_page`` can be set accurately.

Public API
----------
build_sections(json_path, images_source_dir, output_dir, book_name) -> list[dict]
"""

from __future__ import annotations

import base64
import json
import re
import shutil
from pathlib import Path
from typing import Any


# ═══════════════════════════════════════════════════════════════════════════
# Slugification — matches split_sections.py's _slugify() exactly
# ═══════════════════════════════════════════════════════════════════════════

def _slugify(title: str, max_length: int = 60) -> str:
    """Convert a human-readable title into a filesystem-safe slug.

    This is intentionally identical to the ``_slugify`` in
    ``split_sections.py`` so that section IDs are consistent in style
    across both preprocessing paths. This matters if a book series mixes
    scanned and text-based volumes.

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


# ═══════════════════════════════════════════════════════════════════════════
# Block-tree traversal helpers
# ═══════════════════════════════════════════════════════════════════════════

def _extract_page_number(block_id: str) -> int | None:
    """Extract the 0-indexed page number from a Marker block ID.

    Marker block IDs follow the pattern ``/page/<N>/...`` where ``<N>``
    is a 0-indexed page number.

    Parameters
    ----------
    block_id : str
        The block's ``id`` field from Marker's JSON output.

    Returns
    -------
    int or None
        The page number if found, else None.

    Examples
    --------
    >>> _extract_page_number("/page/0/Page/1/SectionHeader/2")
    0
    >>> _extract_page_number("/page/42/Page/1/Text/5")
    42
    """
    match = re.search(r"/page/(\d+)", block_id)
    if match:
        return int(match.group(1))
    return None


def _extract_heading_level(block: dict) -> int:
    """Determine the heading level from a SectionHeader block.

    Marker may encode the heading level in several ways:
    - An ``html`` field containing ``<h1>``, ``<h2>``, ``<h3>`` tags
    - A ``heading_level`` field (some versions)
    - The block_type itself may include level info

    We check all of these and default to level 2 (sub-section) if
    we can't determine the level.

    Parameters
    ----------
    block : dict
        A Marker block with ``block_type == "SectionHeader"``.

    Returns
    -------
    int
        Heading level (1 = chapter, 2 = section, 3 = sub-section).
    """
    # Check for explicit heading_level field
    if "heading_level" in block:
        return int(block["heading_level"])

    # Check HTML content for heading tags (<h1>, <h2>, <h3>)
    html = block.get("html", "")
    h_match = re.search(r"<h(\d)", html)
    if h_match:
        return int(h_match.group(1))

    # Default to level 2 (section) if we can't determine
    return 2


def _extract_text_from_html(html: str) -> str:
    """Strip HTML tags to get plain text content.

    This is a lightweight tag-stripper — not a full HTML parser, but
    sufficient for Marker's relatively simple HTML output.

    Parameters
    ----------
    html : str
        HTML string from a Marker block's ``html`` field.

    Returns
    -------
    str
        Plain text with HTML tags removed.
    """
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", "", html)
    # Decode common HTML entities
    text = text.replace("&amp;", "&")
    text = text.replace("&lt;", "<")
    text = text.replace("&gt;", ">")
    text = text.replace("&quot;", '"')
    text = text.replace("&#39;", "'")
    text = text.replace("&nbsp;", " ")
    # Collapse excessive whitespace but preserve paragraph breaks
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _save_image_from_block(
    block: dict,
    images_source_dir: Path,
    images_dest_dir: Path,
    section_id: str,
    counter: int,
    prefix: str = "img",
) -> str | None:
    """Save an image associated with a Marker block to the images directory.

    Marker stores images in two possible ways:
    1. As files in the images/ subdirectory of the output folder
    2. As base64-encoded data in the block's ``images`` dict

    Parameters
    ----------
    block : dict
        The Marker block containing image data.
    images_source_dir : Path
        Directory where Marker saved its extracted images.
    images_dest_dir : Path
        Destination ``images/`` directory inside the book's processed folder.
    section_id : str
        Current section's ID, used for naming the saved file.
    counter : int
        Running counter for unique filenames.
    prefix : str
        Filename prefix — "img" for figures, "eq" for equations.

    Returns
    -------
    str or None
        The filename of the saved image (relative to images_dest_dir),
        or None if no image could be saved.
    """
    # Strategy 1: Check the block's "images" dict for base64-encoded data
    block_images = block.get("images", {})
    if block_images and isinstance(block_images, dict):
        for img_name, img_data in block_images.items():
            # Determine file extension from the image name
            ext = Path(img_name).suffix if "." in img_name else ".png"
            filename = f"{section_id}_{prefix}_{counter}{ext}"
            dest_path = images_dest_dir / filename

            # img_data might be base64-encoded
            if isinstance(img_data, str):
                try:
                    # Try decoding as base64
                    image_bytes = base64.b64decode(img_data)
                    dest_path.write_bytes(image_bytes)
                    return filename
                except Exception:
                    pass

            # img_data might be raw bytes
            if isinstance(img_data, bytes):
                dest_path.write_bytes(img_data)
                return filename

    # Strategy 2: Check if there's a referenced image file in the source dir
    # Marker may reference images by filename in the HTML content
    html = block.get("html", "")
    img_match = re.search(r'src="([^"]+)"', html)
    if img_match:
        img_ref = img_match.group(1)
        # Try to find the referenced image in the source directory
        source_candidates = [
            images_source_dir / img_ref,
            images_source_dir / Path(img_ref).name,
            images_source_dir / "images" / Path(img_ref).name,
        ]
        for source_path in source_candidates:
            if source_path.exists() and source_path.is_file():
                ext = source_path.suffix or ".png"
                filename = f"{section_id}_{prefix}_{counter}{ext}"
                dest_path = images_dest_dir / filename
                shutil.copy2(str(source_path), str(dest_path))
                return filename

    return None


# ═══════════════════════════════════════════════════════════════════════════
# Block-type classification
# ═══════════════════════════════════════════════════════════════════════════

# Block types that indicate a section header
_HEADER_TYPES = {"SectionHeader"}

# Block types that contain text content to append to a section
_TEXT_TYPES = {"Text", "TextInlineMath", "ListItem", "ListGroup",
               "Caption", "Footnote", "Code"}

# Block types that contain images / figures
_IMAGE_TYPES = {"Picture", "Figure"}

# Block types that contain equations
_EQUATION_TYPES = {"Equation"}

# Block types that contain tables (rendered as text)
_TABLE_TYPES = {"Table"}


# ═══════════════════════════════════════════════════════════════════════════
# Core: flatten the block tree into document-order blocks
# ═══════════════════════════════════════════════════════════════════════════

def _flatten_blocks(data: Any) -> list[dict]:
    """Recursively flatten Marker's hierarchical block tree into a flat
    list in document reading order.

    Marker's JSON can be structured in several ways depending on version:
    - A list of page-level blocks, each with ``children``
    - A dict with a ``children`` key at the top level
    - A list of blocks directly

    This function handles all variants and produces a flat list where
    each block has at minimum ``block_type``, ``html``, and ``id``.

    Parameters
    ----------
    data : dict or list
        The parsed JSON from Marker's output file.

    Returns
    -------
    list[dict]
        Flat list of blocks in document order.
    """
    flat: list[dict] = []

    def _walk(node: Any) -> None:
        """Recursively walk a node and collect leaf-level content blocks."""
        if isinstance(node, list):
            for item in node:
                _walk(item)
        elif isinstance(node, dict):
            block_type = node.get("block_type", node.get("type", ""))
            # If this node has content, add it to the flat list
            if block_type:
                flat.append(node)
            # Recurse into children
            children = node.get("children", [])
            if children:
                _walk(children)

    _walk(data)
    return flat


# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════

def build_sections(
    json_path: str,
    images_source_dir: str,
    output_dir: str,
    book_name: str,
) -> list[dict]:
    """Parse Marker's JSON output and produce ``sections/*.json`` files
    matching the exact output contract used by the text-based path.

    This is the function that makes ``build_index.py``, ``core/router.py``,
    and ``core/generator.py`` completely agnostic to which preprocessing
    path produced a given book.

    Parameters
    ----------
    json_path : str
        Path to Marker's JSON output file.
    images_source_dir : str
        Path to the directory where Marker saved extracted images.
    output_dir : str
        Base output directory for this book
        (e.g. ``books/processed/fluid_mechanics``).
    book_name : str
        Human-readable book name (used only for logging).

    Returns
    -------
    list[dict]
        Lightweight section-metadata dicts (same shape as
        ``split_sections.py``'s return value), suitable for downstream
        stages like ``build_index.py``.
    """
    json_path_obj = Path(json_path)
    images_source_dir_obj = Path(images_source_dir)
    output_path = Path(output_dir)
    sections_dir = output_path / "sections"
    images_dest_dir = output_path / "images"

    # Create output directories
    sections_dir.mkdir(parents=True, exist_ok=True)
    images_dest_dir.mkdir(parents=True, exist_ok=True)

    # ── Load and flatten Marker's block tree ─────────────────────────────
    raw_data = json.loads(json_path_obj.read_text(encoding="utf-8"))
    flat_blocks = _flatten_blocks(raw_data)

    print(
        f"[MARKER→SECTIONS] Loaded {len(flat_blocks)} blocks from "
        f"'{json_path_obj.name}'"
    )

    # ── Walk blocks in document order, accumulating sections ─────────────
    #
    # State machine:
    #   - When we hit a SectionHeader, close the current section and start
    #     a new one.
    #   - Text/Table blocks get appended to the current section's text.
    #   - Picture/Figure blocks get saved as images.
    #   - Equation blocks get saved as equations (distinct from images!).
    #   - Content before the first header goes into a "front-matter" section.

    all_sections: list[dict[str, Any]] = []
    seen_slugs: dict[str, int] = {}           # handle duplicate slugs

    # Accumulators for the current section being built
    current_title: str = "Front Matter"
    current_level: int = 1
    current_text_parts: list[str] = []
    current_images: list[dict[str, Any]] = []
    current_equations: list[dict[str, Any]] = []
    current_start_page: int = 0
    current_end_page: int = 0
    current_chapter: str = "Front Matter"     # most recent level-1 heading

    # Counters for unique image filenames within each section
    img_counter: int = 0
    eq_counter: int = 0

    def _close_current_section() -> None:
        """Finalize and save the current section accumulator."""
        nonlocal current_title, current_text_parts, current_images
        nonlocal current_equations, current_start_page, current_end_page
        nonlocal img_counter, eq_counter

        # Build the section text
        full_text = "\n".join(current_text_parts).strip()

        # Skip completely empty front-matter sections
        if (current_title == "Front Matter"
                and not full_text
                and not current_images
                and not current_equations):
            return

        # Generate a unique slug for the section ID
        slug = _slugify(current_title)
        if not slug:
            slug = "untitled"
        if slug in seen_slugs:
            seen_slugs[slug] += 1
            slug = f"{slug}-{seen_slugs[slug]}"
        else:
            seen_slugs[slug] = 0

        section_doc: dict[str, Any] = {
            "section_id": slug,
            "title": current_title,
            "level": current_level,
            "chapter": current_chapter,
            "parent_id": None,               # populated in hierarchy pass
            "children": [],                   # populated in hierarchy pass
            "start_page": current_start_page,
            "end_page": current_end_page,
            "text": full_text,
            "images": current_images,
            "equations": current_equations,
        }

        all_sections.append(section_doc)

        # Reset accumulators for the next section
        current_text_parts = []
        current_images = []
        current_equations = []
        img_counter = 0
        eq_counter = 0

    # ── Main block-walking loop ──────────────────────────────────────────
    for block in flat_blocks:
        block_type = block.get("block_type", block.get("type", ""))
        block_id = block.get("id", "")
        html = block.get("html", "")

        # Track page numbers from block IDs
        page_num = _extract_page_number(block_id)
        if page_num is not None:
            current_end_page = max(current_end_page, page_num)

        # ── SectionHeader: close current section, start new one ──────
        if block_type in _HEADER_TYPES:
            _close_current_section()

            current_title = _extract_text_from_html(html)
            current_level = _extract_heading_level(block)
            current_start_page = page_num if page_num is not None else current_end_page
            current_end_page = current_start_page

            # Update the running chapter name (most recent level-1 heading)
            if current_level == 1:
                current_chapter = current_title

            print(
                f"[MARKER→SECTIONS]   Section: '{current_title}' "
                f"(level {current_level}, page {current_start_page})"
            )

        # ── Text content blocks ──────────────────────────────────────
        elif block_type in _TEXT_TYPES:
            text = _extract_text_from_html(html)
            if text:
                current_text_parts.append(text)

        # ── Table blocks — render as text ────────────────────────────
        elif block_type in _TABLE_TYPES:
            # Tables are rendered as HTML by Marker.  We extract the
            # text content, which loses the table structure but preserves
            # the data for search/indexing purposes.
            text = _extract_text_from_html(html)
            if text:
                current_text_parts.append(f"\n[Table]\n{text}\n")

        # ── Image / Figure blocks ────────────────────────────────────
        elif block_type in _IMAGE_TYPES:
            img_counter += 1
            slug = _slugify(current_title) or "untitled"
            filename = _save_image_from_block(
                block, images_source_dir_obj, images_dest_dir,
                slug, img_counter, prefix="img"
            )
            if filename:
                current_images.append({
                    "path": f"images/{filename}",
                    "type": "figure",
                    "caption": _extract_text_from_html(html) or None,
                })

        # ── Equation blocks ──────────────────────────────────────────
        elif block_type in _EQUATION_TYPES:
            # First, add the equation text to the section body
            eq_text = _extract_text_from_html(html)
            if eq_text:
                current_text_parts.append(eq_text)

            # Then, try to save the equation as an image
            eq_counter += 1
            slug = _slugify(current_title) or "untitled"
            filename = _save_image_from_block(
                block, images_source_dir_obj, images_dest_dir,
                slug, eq_counter, prefix="eq"
            )
            if filename:
                current_equations.append({
                    "path": f"images/{filename}",
                    "type": "equation",
                    "label": None,
                })

        # ── Other block types (Page, PageHeader, PageFooter, etc.) ───
        # Silently skip — these are structural blocks, not content.

    # Close the final section
    _close_current_section()

    # ── Build hierarchy (parent_id / children) ───────────────────────────
    #
    # Walk through sections in order.  For each section, its parent is the
    # most recent preceding section with a *lower* level number.
    # Level 1 sections (chapters) have no parent.

    id_to_section: dict[str, dict] = {s["section_id"]: s for s in all_sections}

    for i, sec in enumerate(all_sections):
        if sec["level"] <= 1:
            sec["parent_id"] = None
            continue

        # Search backwards for the nearest section with a lower level
        for j in range(i - 1, -1, -1):
            if all_sections[j]["level"] < sec["level"]:
                sec["parent_id"] = all_sections[j]["section_id"]
                # Add this section as a child of its parent
                parent = id_to_section[all_sections[j]["section_id"]]
                if sec["section_id"] not in parent["children"]:
                    parent["children"].append(sec["section_id"])
                break

    # ── Write section JSONs ──────────────────────────────────────────────
    for sec in all_sections:
        json_out_path = sections_dir / f"{sec['section_id']}.json"
        json_out_path.write_text(
            json.dumps(sec, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # ── Summary ──────────────────────────────────────────────────────────
    total_images = sum(len(s["images"]) for s in all_sections)
    total_equations = sum(len(s["equations"]) for s in all_sections)

    print(
        f"[MARKER→SECTIONS] Created {len(all_sections)} sections "
        f"({total_images} images, {total_equations} equations) "
        f"for '{book_name}'"
    )

    # ── Return lightweight metadata (same shape as split_sections.py) ────
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
# CLI entry-point — useful for standalone testing
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 4:
        print(
            "Usage: python -m preprocessing.parse_marker_output "
            "<marker_json> <marker_images_dir> <output_dir> [book_name]\n"
            "\n"
            "Parses Marker's JSON output and produces sections/*.json files\n"
            "matching the exact schema used by the text-based path.\n"
            "Run this after marker_ocr.py to complete the Marker pipeline."
        )
        sys.exit(1)

    _json_path = sys.argv[1]
    _images_dir = sys.argv[2]
    _output_dir = sys.argv[3]
    _book_name = sys.argv[4] if len(sys.argv) > 4 else "Unknown Book"

    result = build_sections(_json_path, _images_dir, _output_dir, _book_name)

    print(f"\n{'─' * 60}")
    print(f"  Parse Result: {len(result)} sections")
    print(f"{'─' * 60}")
    for sec in result:
        parent = f" (parent: {sec['parent_id']})" if sec["parent_id"] else ""
        kids = f" [{len(sec['children'])} children]" if sec["children"] else ""
        print(
            f"  • [{sec['section_id']}] {sec['title']} "
            f"(pp. {sec['start_page']}–{sec['end_page']}){parent}{kids}"
        )
