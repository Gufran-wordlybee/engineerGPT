"""
run_pipeline.py — Orchestrate the full Phase 0 preprocessing pipeline.

Runs TOC extraction → section splitting → image extraction → index building
for a single PDF or every PDF found in the raw books directory.

Usage
-----
::

    # Process a single book
    python -m preprocessing.run_pipeline books/raw/fluid_mechanics.pdf

    # Process all PDFs in books/raw/
    python -m preprocessing.run_pipeline

    # Re-process even if output already exists
    python -m preprocessing.run_pipeline --force
    python -m preprocessing.run_pipeline books/raw/fluid_mechanics.pdf --force
"""

import sys
import time
from pathlib import Path
from typing import Optional

from config.settings import BOOKS_RAW_PATH, BOOKS_PROCESSED_PATH

from preprocessing.extract_toc import extract_toc
from preprocessing.split_sections import split_sections
from preprocessing.extract_images import extract_images
from preprocessing.build_index import build_index


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
# Core pipeline
# ---------------------------------------------------------------------------

def process_book(pdf_path: str, *, force: bool = False) -> None:
    """Run the full preprocessing pipeline for a single book.

    Parameters
    ----------
    pdf_path : str
        Path to the source PDF file.
    force : bool, optional
        If ``True``, reprocess the book even when output already exists.
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

    # 4. Step 1 — Extract TOC
    print("\n[Step 1/4] Extracting TOC …")
    toc = extract_toc(pdf_path)
    if not toc:
        print(
            f"[PIPELINE] Warning: No TOC found for '{human_name}'. "
            f"Skipping this book."
        )
        return

    # 5. Step 2 — Split sections
    print("[Step 2/4] Splitting sections …")
    sections = split_sections(pdf_path, toc, str(output_dir))

    # 6. Step 3 — Extract images
    print("[Step 3/4] Extracting images …")
    image_count = extract_images(pdf_path, str(output_dir), sections)

    # 7. Step 4 — Build index
    print("[Step 4/4] Building index …")
    index = build_index(human_name, str(output_dir))

    # 8. Summary
    elapsed = time.time() - start_time
    section_count = index.get("total_sections", len(sections))

    print(f"\n  Summary for '{human_name}':")
    print(f"    Sections : {section_count}")
    print(f"    Images   : {image_count}")
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
            process_book(str(pdf_path), force=force)
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
