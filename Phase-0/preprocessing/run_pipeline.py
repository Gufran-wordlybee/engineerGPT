"""
run_pipeline.py — Orchestrate the full Phase 0 preprocessing pipeline.

Supports **two** preprocessing paths that converge on the same output:

1. **Text-based PDFs** (existing path):
   TOC extraction → section splitting → image extraction → index building
   Uses pymupdf to directly extract text, headings, and images.

2. **Scanned PDFs** (new Marker OCR path):
   Scanned detection → Marker OCR → parse Marker output → index building
   Uses the Marker library for layout-aware OCR on image-only PDFs.

Both paths produce identical output in ``books/processed/<book>/sections/``
and ``books/processed/<book>/images/``, so ``build_index.py``,
``core/router.py``, and ``core/generator.py`` require **zero changes**
regardless of which path produced a given book.

The fork decision is made by ``scanned_detector.classify_pdf()``, which
samples pages and checks if they have extractable text. You can override
the auto-detection with ``--force-scanned`` or ``--force-text``.

Usage
-----
::

    # Process a single book (auto-detects scanned vs text)
    python -m preprocessing.run_pipeline books/raw/fluid_mechanics.pdf

    # Process all PDFs in books/raw/
    python -m preprocessing.run_pipeline

    # Re-process even if output already exists
    python -m preprocessing.run_pipeline --force

    # Force a specific preprocessing path (bypass auto-detection)
    python -m preprocessing.run_pipeline books/raw/old_scan.pdf --force-scanned
    python -m preprocessing.run_pipeline books/raw/modern_book.pdf --force-text
"""

import sys
import time
from pathlib import Path
from typing import Optional

from config.settings import BOOKS_RAW_PATH, BOOKS_PROCESSED_PATH

# Existing text-based path imports
from preprocessing.extract_toc import extract_toc
from preprocessing.split_sections import split_sections
from preprocessing.extract_images import extract_images
from preprocessing.build_index import build_index

# New scanned-PDF path imports
from preprocessing.scanned_detector import classify_pdf
from preprocessing.marker_ocr import run_marker
from preprocessing.parse_marker_output import build_sections


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _derive_book_name(pdf_path: str) -> str:
    """Derive a normalised book name from a PDF file path.

    Takes the file stem (no extension), replaces hyphens with underscores,
    and lowercases the result.

    Examples
    --------
    >>> _derive_book_name("books/raw/Fluid-Mechanics.pdf")
    'fluid_mechanics'
    """
    stem = Path(pdf_path).stem
    return stem.replace("-", "_").lower()


def _human_readable_name(book_name: str) -> str:
    """Convert a snake_case book name to a human-readable, title-cased string.

    Examples
    --------
    >>> _human_readable_name("fluid_mechanics")
    'Fluid Mechanics'
    """
    return book_name.replace("_", " ").replace("-", " ").title()


# ---------------------------------------------------------------------------
# Text-based pipeline path (existing, untouched logic)
# ---------------------------------------------------------------------------

def _run_text_path(
    pdf_path: str,
    human_name: str,
    output_dir: Path,
) -> tuple[int, dict]:
    """Run the original pymupdf-based pipeline for text-based PDFs.

    Steps: TOC extraction → section splitting → image extraction → index build

    Parameters
    ----------
    pdf_path : str
        Path to the source PDF.
    human_name : str
        Human-readable book name for display.
    output_dir : Path
        Output directory for this book.

    Returns
    -------
    tuple[int, dict]
        (visual_count, index_dict) — number of extracted visuals and the
        built index.
    """
    # Step 1 — Extract TOC
    print("\n[Step 1/4] Extracting TOC …")
    toc = extract_toc(pdf_path)
    if not toc:
        print(
            f"[PIPELINE] Warning: No TOC found for '{human_name}'. "
            f"Skipping this book."
        )
        return 0, {}

    # Step 2 — Split sections (with hierarchy, cleanup, QA)
    print("[Step 2/4] Splitting sections …")
    sections = split_sections(pdf_path, toc, str(output_dir))

    # Step 3 — Extract visuals (raster images + vector diagrams + equations)
    print("[Step 3/4] Extracting visuals …")
    visual_count = extract_images(pdf_path, str(output_dir), sections)

    # Step 4 — Build index (hierarchical, TF-IDF, abstracts)
    print("[Step 4/4] Building index …")
    index = build_index(human_name, str(output_dir))

    return visual_count, index


# ---------------------------------------------------------------------------
# Scanned-PDF pipeline path (new Marker OCR path)
# ---------------------------------------------------------------------------

def _run_scanned_path(
    pdf_path: str,
    human_name: str,
    output_dir: Path,
    detection_result: dict | None = None,
) -> tuple[int, dict]:
    """Run the Marker OCR pipeline for scanned PDFs.

    Steps: Marker OCR (chunked) → parse output → index build

    The Marker path produces the exact same ``sections/*.json`` and
    ``images/`` structure as the text-based path, so ``build_index.py``
    works identically on the result.

    Parameters
    ----------
    pdf_path : str
        Path to the source PDF.
    human_name : str
        Human-readable book name for display.
    output_dir : Path
        Output directory for this book.
    detection_result : dict, optional
        Result from ``classify_pdf()`` — contains ``needs_force_ocr``
        which tells us whether this is a true image-only scan or a
        partial-OCR scan with some usable text layer.

    Returns
    -------
    tuple[int, dict]
        (visual_count, index_dict) — number of extracted visuals and the
        built index.
    """
    # Step 1 — Run Marker OCR (chunked — see marker_ocr.py for details)
    #
    # Marker writes its raw output to a 'marker_raw/' subfolder.
    # Inside that, each chunk gets its own subdirectory, and a 'merged/'
    # directory holds the combined output.
    marker_raw_dir = output_dir / "marker_raw"
    marker_raw_dir.mkdir(parents=True, exist_ok=True)

    # Smart force-OCR decision:
    #   - If detection_result says needs_force_ocr=True → pure image scan,
    #     force Marker to re-OCR every page (slow but necessary).
    #   - If needs_force_ocr=False → partial text layer exists, let Marker
    #     use it where it can (30–70% faster).
    #   - If no detection_result (e.g. --force-scanned), fall back to the
    #     global MARKER_FORCE_OCR setting.
    smart_force_ocr = None  # let marker_ocr use global setting
    if detection_result is not None:
        smart_force_ocr = detection_result.get("needs_force_ocr", None)
        ocr_label = "force-OCR" if smart_force_ocr else "partial-OCR"
        avg = detection_result.get("avg_chars_per_page", "?")
        print(f"[PIPELINE] Smart OCR decision: {ocr_label} (avg {avg} chars/page)")

    print("\n[Step 1/3] Running Marker OCR (chunked) …")
    marker_result = run_marker(
        pdf_path,
        str(marker_raw_dir),
        force_ocr=smart_force_ocr,
    )

    # Step 2 — Parse Marker's merged output into sections/*.json
    print("[Step 2/3] Parsing Marker output into sections …")
    sections = build_sections(
        json_path=marker_result["json_path"],
        images_source_dir=marker_result["images_dir"],
        output_dir=str(output_dir),
        book_name=human_name,
    )

    # Count total visuals for the summary
    visual_count = 0
    sections_dir = output_dir / "sections"
    for json_file in sections_dir.glob("*.json"):
        import json
        sec_data = json.loads(json_file.read_text(encoding="utf-8"))
        visual_count += len(sec_data.get("images", []))
        visual_count += len(sec_data.get("equations", []))

    # Step 3 — Build index (same build_index.py, completely unmodified)
    print("[Step 3/3] Building index …")
    index = build_index(human_name, str(output_dir))

    return visual_count, index


# ---------------------------------------------------------------------------
# Core pipeline — the fork point
# ---------------------------------------------------------------------------

def process_book(
    pdf_path: str,
    *,
    force: bool = False,
    force_path: str | None = None,
) -> None:
    """Run the full preprocessing pipeline for a single book.

    Automatically detects whether the PDF is scanned or text-based, then
    routes it through the appropriate preprocessing path.

    Parameters
    ----------
    pdf_path : str
        Path to the source PDF file.
    force : bool, optional
        If ``True``, reprocess the book even when output already exists.
    force_path : str, optional
        Override auto-detection: ``"scanned"`` or ``"text"``.  If not set,
        the detector decides.
    """
    start_time = time.time()

    # 1. Derive identifiers
    book_name = _derive_book_name(pdf_path)
    human_name = _human_readable_name(book_name)
    output_dir = Path(BOOKS_PROCESSED_PATH) / book_name

    # 2. Check if already processed
    index_file = output_dir / "index.json"
    if index_file.exists() and not force:
        print(
            f"[PIPELINE] Skipping '{human_name}' — already processed. "
            f"Use --force to reprocess."
        )
        return

    # 3. Banner
    print(f"\n{'=' * 50}")
    print(f"  Processing: {human_name}")
    print(f"{'=' * 50}")

    # 4. Classify the PDF — scanned or text-based?
    #
    # detection_result holds the output of scanned_detector.classify_pdf(),
    # including the crucial `needs_force_ocr` flag.  It's None when the
    # user overrides detection with --force-scanned / --force-text.
    detection_result: dict | None = None

    if force_path:
        # Manual override — skip auto-detection
        classification = force_path
        print(f"[PIPELINE] Using forced path: '{classification}'")
    else:
        # Auto-detect using scanned_detector
        detection_result = classify_pdf(pdf_path)
        classification = detection_result["classification"]

    # 5. Run the appropriate pipeline path
    if classification == "scanned":
        print(f"[PIPELINE] → Scanned path (Marker OCR)")
        visual_count, index = _run_scanned_path(
            pdf_path, human_name, output_dir,
            detection_result=detection_result,   # ← pass smart OCR decision
        )
        path_label = "Marker OCR"
    else:
        print(f"[PIPELINE] → Text path (pymupdf)")
        visual_count, index = _run_text_path(
            pdf_path, human_name, output_dir
        )
        path_label = "pymupdf"

    # 6. Summary
    elapsed = time.time() - start_time
    section_count = index.get("total_sections", 0)

    print(f"\n  Summary for '{human_name}':")
    print(f"    Path     : {path_label}")
    print(f"    Sections : {section_count}")
    print(f"    Visuals  : {visual_count}")
    print(f"    Output   : {output_dir}")
    print(f"    Elapsed  : {elapsed:.1f}s")
    print(f"{'=' * 50}")
    print(f"  Done: {human_name}")
    print(f"{'=' * 50}\n")


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point.  Process a specific PDF or all PDFs in the raw books folder."""
    args = sys.argv[1:]

    # Parse flags
    force: bool = "--force" in args
    if force:
        args.remove("--force")

    # --force-scanned / --force-text override auto-detection
    force_path: str | None = None
    if "--force-scanned" in args:
        force_path = "scanned"
        args.remove("--force-scanned")
    elif "--force-text" in args:
        force_path = "text"
        args.remove("--force-text")

    # Determine which PDF(s) to process
    pdf_paths: list[Path] = []

    if args:
        # A specific path was provided
        target = Path(args[0])
        if not target.exists():
            print(f"[PIPELINE] Error: file not found — {target}")
            sys.exit(1)
        if not target.suffix.lower() == ".pdf":
            print(f"[PIPELINE] Error: expected a .pdf file — {target}")
            sys.exit(1)
        pdf_paths.append(target)
    else:
        # Discover all PDFs in the raw books directory
        raw_dir = Path(BOOKS_RAW_PATH)
        if not raw_dir.is_dir():
            print(f"[PIPELINE] Error: raw books directory not found — {raw_dir}")
            sys.exit(1)
        pdf_paths = sorted(raw_dir.glob("*.pdf"))
        if not pdf_paths:
            print(f"[PIPELINE] No PDF files found in {raw_dir}")
            sys.exit(0)

    # Process each book
    total_start = time.time()
    succeeded = 0
    failed = 0

    for pdf_path in pdf_paths:
        try:
            process_book(str(pdf_path), force=force, force_path=force_path)
            succeeded += 1
        except Exception as exc:
            print(f"[PIPELINE] Error processing {pdf_path.name}: {exc}")
            failed += 1

    # Final report
    total_elapsed = time.time() - total_start
    print(f"\n[PIPELINE] Finished — {succeeded} succeeded, {failed} failed, "
          f"total time {total_elapsed:.1f}s")


if __name__ == "__main__":
    main()
