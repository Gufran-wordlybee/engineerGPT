"""
preprocessing.extract_images
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Extracts images from each section's page range in a PDF and updates the
corresponding section JSON files with relative image paths.

Images that fall below the minimum size threshold (``MIN_IMAGE_SIZE`` from
``config.settings``) are discarded, as they are typically decorative icons
or scanning artifacts.  Duplicate images (same xref across pages) are
extracted only once.

Public API
----------
extract_images(pdf_path, output_dir, sections) -> int
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import pymupdf  # PyMuPDF

from config.settings import MIN_IMAGE_SIZE


# ─── public API ─────────────────────────────────────────────────────────────


def extract_images(
    pdf_path: str,
    output_dir: str,
    sections: list[dict],
) -> int:
    """Extract images from every section's page range and update section JSONs.

    Parameters
    ----------
    pdf_path : str
        Filesystem path to the source PDF.
    output_dir : str
        Base output directory for this book (e.g.
        ``books/processed/fluid_mechanics/``).
    sections : list[dict]
        Section metadata dicts as returned by
        :func:`preprocessing.split_sections.split_sections`.  Each dict must
        contain at least ``section_id``, ``title``, ``start_page``, and
        ``end_page``.

    Returns
    -------
    int
        Total number of images extracted across all sections.
    """
    output_path = Path(output_dir)
    images_dir = output_path / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    sections_dir = output_path / "sections"

    doc: pymupdf.Document = pymupdf.open(str(pdf_path))
    total_images: int = 0

    # Track xrefs globally so the same embedded image is never saved twice.
    seen_xrefs: set[int] = set()

    for section in sections:
        section_id: str = section["section_id"]
        title: str = section["title"]
        start_page: int = section["start_page"]
        end_page: int = section["end_page"]

        section_image_paths: list[str] = []
        img_counter: int = 0

        for page_num in range(start_page, end_page + 1):
            page: pymupdf.Page = doc[page_num]
            image_list: list[tuple] = page.get_images(full=True)

            for img_info in image_list:
                xref: int = img_info[0]

                if xref in seen_xrefs:
                    continue
                seen_xrefs.add(xref)

                # ── attempt extraction ──────────────────────────────────
                try:
                    base_image: dict = doc.extract_image(xref)
                except Exception as exc:
                    warnings.warn(
                        f"[IMAGES] Could not extract xref {xref} on "
                        f"page {page_num} – skipping ({exc})",
                        stacklevel=2,
                    )
                    continue

                if base_image is None:
                    continue

                width: int = base_image.get("width", 0)
                height: int = base_image.get("height", 0)

                # Skip tiny images (icons, 1-px spacers, etc.).
                if width < MIN_IMAGE_SIZE or height < MIN_IMAGE_SIZE:
                    continue

                # Note CMYK colour-space if present (PyMuPDF's
                # extract_image usually converts to RGB automatically, but
                # downstream consumers may want to know).
                colorspace: int = base_image.get("colorspace", 0)
                if colorspace == 4:  # CMYK
                    warnings.warn(
                        f"[IMAGES] xref {xref} uses CMYK colour-space; "
                        f"extracted bytes may still be CMYK.",
                        stacklevel=2,
                    )

                ext: str = base_image.get("ext", "png")
                image_bytes: bytes = base_image["image"]

                img_counter += 1
                filename = f"{section_id}_img_{img_counter}.{ext}"
                dest_path: Path = images_dir / filename
                dest_path.write_bytes(image_bytes)

                # Store the path *relative* to output_dir for portability.
                relative_path: str = str(
                    dest_path.relative_to(output_path)
                )
                section_image_paths.append(relative_path)

        # ── update the section JSON with image paths ────────────────────
        json_path: Path = sections_dir / f"{section_id}.json"
        if json_path.exists():
            section_doc: dict = json.loads(
                json_path.read_text(encoding="utf-8")
            )
            section_doc["images"] = section_image_paths
            json_path.write_text(
                json.dumps(section_doc, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        section_count = len(section_image_paths)
        total_images += section_count
        print(
            f"[IMAGES] Section '{title}': extracted {section_count} images"
        )

    doc.close()
    return total_images


# ─── standalone entry point ─────────────────────────────────────────────────

if __name__ == "__main__":
    print(
        "Usage:\n"
        "  python -m preprocessing.extract_images\n\n"
        "This module is not intended to be run directly.  Use it as part\n"
        "of the preprocessing pipeline:\n\n"
        "  from preprocessing.split_sections import split_sections\n"
        "  from preprocessing.extract_images import extract_images\n\n"
        "  sections = split_sections(pdf_path, toc, output_dir)\n"
        "  total = extract_images(pdf_path, output_dir, sections)\n"
        "  print(f'Extracted {total} images.')"
    )
