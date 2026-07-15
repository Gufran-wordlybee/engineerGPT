"""
preprocessing.marker_ocr
~~~~~~~~~~~~~~~~~~~~~~~~~
Run the **Marker** CLI (``marker_single``) against a scanned PDF to produce
layout-aware OCR output (JSON blocks + extracted images).

Architecture
------------
This module sits between ``scanned_detector.py`` (which decides *if* we
need Marker) and ``parse_marker_output.py`` (which translates Marker's
output into the standard ``sections/*.json`` schema).

    scanned_detector → **marker_ocr** → parse_marker_output → build_index

This module's responsibility ends at producing Marker's raw output on disk.
It does **not** know anything about the downstream ``sections/*.json`` schema.

Chunked Processing (the key improvement)
-----------------------------------------
Instead of processing the entire PDF in one ``marker_single`` call (which
means losing ALL progress on crash/timeout/Cmd+C), we split the book into
page-range chunks and process each chunk separately:

1. Get total page count via pymupdf.
2. Split into chunks of ``MARKER_CHUNK_SIZE_PAGES`` (default 50 pages).
3. For each chunk:
   - **Skip if already completed** (resume support — re-running after a
     crash picks up exactly where it left off).
   - Call ``marker_single`` with ``--page_range "start-end"``
   - Enforce a per-chunk timeout (``MARKER_CHUNK_TIMEOUT_SECONDS``)
   - On timeout: log a warning and move to the next chunk, rather than
     killing the entire book's progress.
4. After all chunks: merge the per-chunk JSON outputs into one combined
   JSON file that ``parse_marker_output.py`` can consume.

This turns "lose 3 hours of progress on Cmd+C" into "lose at most one
~50-page chunk (~5-10 minutes on CPU)."

Why use the CLI instead of the Python API?
------------------------------------------
There is a known bug (datalab-to/marker issue #906) where the Python API's
``output_format`` config for ``json``/``chunks`` is silently ignored and
returns markdown instead. The CLI does not have this bug.

Public API
----------
run_marker(pdf_path, output_dir, ...) -> dict
    Runs Marker (chunked) and returns paths to the merged JSON + images dir.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pymupdf  # for getting page count — already a project dependency

from config.settings import (
    MARKER_CLI_BINARY,
    MARKER_OUTPUT_FORMAT,
    MARKER_FORCE_OCR,
    MARKER_USE_LLM,
    MARKER_CHUNK_SIZE_PAGES,
    MARKER_CHUNK_TIMEOUT_SECONDS,
)


# ═══════════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════════

def _get_page_count(pdf_path: str) -> int:
    """Get total page count from a PDF using pymupdf.

    This is cheap (just reads PDF metadata, doesn't render pages) and
    gives us the page count we need to compute chunk ranges.
    """
    doc = pymupdf.open(pdf_path)
    count = len(doc)
    doc.close()
    return count


def _compute_chunks(total_pages: int, chunk_size: int) -> list[tuple[int, int]]:
    """Split a page range into fixed-size chunks.

    Parameters
    ----------
    total_pages : int
        Total number of pages in the PDF.
    chunk_size : int
        Maximum pages per chunk.

    Returns
    -------
    list[tuple[int, int]]
        List of (start_page, end_page) tuples, both 0-indexed and inclusive.

    Examples
    --------
    >>> _compute_chunks(120, 50)
    [(0, 49), (50, 99), (100, 119)]
    >>> _compute_chunks(30, 50)
    [(0, 29)]
    """
    chunks = []
    for start in range(0, total_pages, chunk_size):
        end = min(start + chunk_size - 1, total_pages - 1)
        chunks.append((start, end))
    return chunks


def _chunk_output_dir(base_dir: Path, start: int, end: int) -> Path:
    """Compute the output subdirectory for a specific chunk.

    Each chunk gets its own subdirectory so that:
    - Marker doesn't overwrite a previous chunk's output
    - We can tell which chunks are "done" (their dir has a JSON file)
    - We can resume from exactly where we left off after a crash
    """
    return base_dir / f"chunk_{start:04d}_{end:04d}"


def _chunk_is_complete(chunk_dir: Path) -> bool:
    """Check if a chunk has already been processed successfully.

    A chunk is considered complete if its output directory exists and
    contains at least one ``.json`` file (Marker's output).

    This is the "resume" mechanism: on re-run after a crash, we skip
    chunks whose output directories already have JSON files in them.
    Only the chunk that was in-progress when the crash happened (which
    will have an empty or partial output dir) gets re-processed.
    """
    if not chunk_dir.exists():
        return False
    json_files = list(chunk_dir.rglob("*.json"))
    return len(json_files) > 0


def _find_json_in_dir(directory: Path) -> Path | None:
    """Find the Marker JSON output file inside a directory.

    Marker creates a subdirectory named after the PDF stem, with the
    JSON file inside it.  This function handles the nesting.
    """
    json_files = list(directory.rglob("*.json"))
    return json_files[0] if json_files else None


def _find_images_in_dir(directory: Path) -> Path:
    """Find the images directory inside a Marker output directory.

    Returns the first ``images/`` subdirectory found, or the directory
    itself as a fallback (Marker sometimes places images loose).
    """
    for candidate in directory.rglob("images"):
        if candidate.is_dir():
            return candidate
    return directory


# ═══════════════════════════════════════════════════════════════════════════
# Single-chunk processing
# ═══════════════════════════════════════════════════════════════════════════

def _run_single_chunk(
    pdf_path: str,
    chunk_output_dir: str,
    start_page: int,
    end_page: int,
    *,
    force_ocr: bool,
    use_llm: bool,
    timeout_seconds: int,
) -> dict[str, Any]:
    """Run ``marker_single`` on a specific page range.

    This is the low-level function that actually calls the Marker CLI.
    It processes one chunk and returns the result.

    Parameters
    ----------
    pdf_path : str
        Path to the source PDF.
    chunk_output_dir : str
        Output directory for this specific chunk.
    start_page, end_page : int
        0-indexed, inclusive page range for this chunk.
    force_ocr : bool
        Whether to force full re-OCR on every page.
    use_llm : bool
        Whether to enable LLM-assisted extraction.
    timeout_seconds : int
        Maximum seconds to wait for this chunk before timing out.

    Returns
    -------
    dict
        ``{"success": bool, "json_path": str|None, "images_dir": str|None,
           "elapsed": float, "error": str|None, "timed_out": bool}``
    """
    chunk_dir = Path(chunk_output_dir)
    chunk_dir.mkdir(parents=True, exist_ok=True)

    # Build the page range string for Marker's --page_range flag
    # Format: "start-end" (0-indexed, inclusive)
    page_range = f"{start_page}-{end_page}"

    # ── Build the marker_single command ──────────────────────────────
    cmd: list[str] = [
        MARKER_CLI_BINARY,
        pdf_path,
        "--output_format", MARKER_OUTPUT_FORMAT,
        "--output_dir", str(chunk_dir),
        "--page_range", page_range,
    ]

    if force_ocr:
        cmd.append("--force_ocr")

    if use_llm:
        cmd.append("--use_llm")

    # ── Execute ──────────────────────────────────────────────────────
    start_time = time.time()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except FileNotFoundError:
        return {
            "success": False,
            "json_path": None,
            "images_dir": None,
            "elapsed": time.time() - start_time,
            "error": (
                f"'{MARKER_CLI_BINARY}' not found. Is marker-pdf installed?\n"
                f"Install with: pip install marker-pdf"
            ),
            "timed_out": False,
        }
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        return {
            "success": False,
            "json_path": None,
            "images_dir": None,
            "elapsed": elapsed,
            "error": (
                f"Chunk pages {start_page}–{end_page} timed out after "
                f"{timeout_seconds}s. Skipping this chunk."
            ),
            "timed_out": True,
        }

    elapsed = time.time() - start_time

    # ── Check for errors ─────────────────────────────────────────────
    if result.returncode != 0:
        stderr_preview = result.stderr[:1000] if result.stderr else "(no stderr)"
        return {
            "success": False,
            "json_path": None,
            "images_dir": None,
            "elapsed": elapsed,
            "error": (
                f"Marker exit code {result.returncode} on pages "
                f"{start_page}–{end_page}:\n{stderr_preview}"
            ),
            "timed_out": False,
        }

    # ── Locate output files ──────────────────────────────────────────
    json_path = _find_json_in_dir(chunk_dir)
    images_dir = _find_images_in_dir(chunk_dir)

    if json_path is None:
        return {
            "success": False,
            "json_path": None,
            "images_dir": None,
            "elapsed": elapsed,
            "error": f"No JSON file found in {chunk_dir} after processing.",
            "timed_out": False,
        }

    return {
        "success": True,
        "json_path": str(json_path),
        "images_dir": str(images_dir),
        "elapsed": elapsed,
        "error": None,
        "timed_out": False,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Chunk merging
# ═══════════════════════════════════════════════════════════════════════════

def _merge_chunk_outputs(
    chunk_dirs: list[Path],
    merged_output_dir: Path,
    pdf_stem: str,
) -> dict[str, str]:
    """Merge per-chunk Marker outputs into a single combined JSON + images dir.

    This is what lets ``parse_marker_output.py`` consume chunked output
    without any changes — it still receives one JSON file containing
    all blocks in document order.

    How merging works
    -----------------
    Each chunk's JSON is a Marker block tree.  We load each one, extract
    the top-level block list (usually under a ``children`` key, or the
    JSON itself is a list), and concatenate them in page order.

    Images from each chunk's output are copied into a single ``images/``
    directory.

    Parameters
    ----------
    chunk_dirs : list[Path]
        Sorted list of chunk output directories that completed successfully.
    merged_output_dir : Path
        Where to write the merged JSON file and combined images directory.
    pdf_stem : str
        The PDF's filename stem, used for naming the merged JSON.

    Returns
    -------
    dict
        ``{"json_path": str, "images_dir": str}``
    """
    merged_output_dir.mkdir(parents=True, exist_ok=True)
    merged_images_dir = merged_output_dir / "images"
    merged_images_dir.mkdir(parents=True, exist_ok=True)

    # Accumulate all blocks from all chunks, in order
    all_blocks: list[dict] = []

    for chunk_dir in chunk_dirs:
        # Find and load this chunk's JSON
        json_path = _find_json_in_dir(chunk_dir)
        if json_path is None:
            continue

        chunk_data = json.loads(json_path.read_text(encoding="utf-8"))

        # Marker's JSON structure varies:
        #   - Could be a dict with "children" key containing block list
        #   - Could be a list of blocks directly
        #   - Could be a dict with other structure
        # We extract the block list regardless of format.
        if isinstance(chunk_data, list):
            all_blocks.extend(chunk_data)
        elif isinstance(chunk_data, dict):
            # Try common keys where blocks live
            if "children" in chunk_data:
                children = chunk_data["children"]
                if isinstance(children, list):
                    all_blocks.extend(children)
                else:
                    all_blocks.append(chunk_data)
            else:
                # Treat the whole dict as a single block (page-level)
                all_blocks.append(chunk_data)

        # Copy images from this chunk's output to the merged images dir
        chunk_images_dir = _find_images_in_dir(chunk_dir)
        if chunk_images_dir.is_dir():
            for img_file in chunk_images_dir.iterdir():
                if img_file.is_file() and not img_file.name.startswith("."):
                    dest = merged_images_dir / img_file.name
                    # Avoid overwriting — prefix with chunk name if conflict
                    if dest.exists():
                        dest = merged_images_dir / f"{chunk_dir.name}_{img_file.name}"
                    shutil.copy2(str(img_file), str(dest))

    # Write the merged JSON — a simple list of all blocks in page order.
    # parse_marker_output.py's _flatten_blocks() handles this format.
    merged_json_path = merged_output_dir / f"{pdf_stem}.json"
    merged_json_path.write_text(
        json.dumps(all_blocks, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(
        f"[MARKER] Merged {len(chunk_dirs)} chunks → "
        f"{len(all_blocks)} top-level blocks in '{merged_json_path.name}'"
    )

    return {
        "json_path": str(merged_json_path),
        "images_dir": str(merged_images_dir),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════

def run_marker(
    pdf_path: str,
    output_dir: str,
    *,
    force_ocr: bool | None = None,
    use_llm: bool | None = None,
    chunk_size: int | None = None,
    chunk_timeout: int | None = None,
) -> dict[str, Any]:
    """Run ``marker_single`` against a PDF using chunked processing.

    The PDF is split into page-range chunks and each chunk is processed
    separately with its own timeout.  Completed chunks are cached to disk,
    so re-running after a crash resumes from where it left off.

    Parameters
    ----------
    pdf_path : str
        Path to the scanned PDF to process.
    output_dir : str
        Directory where Marker should write its output.
    force_ocr : bool, optional
        Override the smart force-OCR decision.  If not set, uses whatever
        the pipeline decided based on ``scanned_detector``'s
        ``needs_force_ocr`` field (passed through by ``run_pipeline.py``).
    use_llm : bool, optional
        Enable LLM-assisted extraction. Defaults to ``MARKER_USE_LLM``.
    chunk_size : int, optional
        Pages per chunk. Defaults to ``MARKER_CHUNK_SIZE_PAGES``.
    chunk_timeout : int, optional
        Timeout per chunk in seconds. Defaults to ``MARKER_CHUNK_TIMEOUT_SECONDS``.

    Returns
    -------
    dict
        ``{"json_path": str, "images_dir": str, "pdf_path": str,
           "chunks_total": int, "chunks_succeeded": int,
           "chunks_failed": int, "chunks_skipped": int,
           "total_elapsed": float}``

    Raises
    ------
    FileNotFoundError
        If *pdf_path* does not exist.
    RuntimeError
        If ALL chunks fail (no usable output at all).
    """
    pdf_path_obj = Path(pdf_path)
    output_dir_obj = Path(output_dir)

    if not pdf_path_obj.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    output_dir_obj.mkdir(parents=True, exist_ok=True)

    # Resolve settings (allow per-call overrides)
    _force_ocr = force_ocr if force_ocr is not None else MARKER_FORCE_OCR
    _use_llm = use_llm if use_llm is not None else MARKER_USE_LLM
    _chunk_size = chunk_size if chunk_size is not None else MARKER_CHUNK_SIZE_PAGES
    _chunk_timeout = chunk_timeout if chunk_timeout is not None else MARKER_CHUNK_TIMEOUT_SECONDS

    pdf_stem = pdf_path_obj.stem
    total_pages = _get_page_count(pdf_path)

    # ── Compute chunks ───────────────────────────────────────────────
    chunks = _compute_chunks(total_pages, _chunk_size)
    total_chunks = len(chunks)

    force_label = "force-OCR" if _force_ocr else "auto-OCR"
    print(
        f"[MARKER] Processing '{pdf_path_obj.name}' "
        f"({total_pages} pages, {total_chunks} chunks of ~{_chunk_size} pages, "
        f"{force_label})"
    )

    # ── Process each chunk ───────────────────────────────────────────
    chunks_dir = output_dir_obj / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    succeeded = 0
    failed = 0
    skipped = 0       # chunks that were already complete from a previous run
    failed_ranges: list[str] = []
    completed_chunk_dirs: list[Path] = []

    total_start = time.time()

    for i, (start, end) in enumerate(chunks, 1):
        chunk_dir = _chunk_output_dir(chunks_dir, start, end)

        # ── Resume support: skip chunks that already have output ─────
        if _chunk_is_complete(chunk_dir):
            print(
                f"[MARKER] Chunk {i}/{total_chunks} (pages {start}–{end}): "
                f"CACHED ✓ (already processed, skipping)"
            )
            completed_chunk_dirs.append(chunk_dir)
            skipped += 1
            continue

        # ── Process this chunk ───────────────────────────────────────
        print(
            f"[MARKER] Chunk {i}/{total_chunks} (pages {start}–{end}): "
            f"processing... (timeout: {_chunk_timeout}s)"
        )

        result = _run_single_chunk(
            pdf_path=pdf_path,
            chunk_output_dir=str(chunk_dir),
            start_page=start,
            end_page=end,
            force_ocr=_force_ocr,
            use_llm=_use_llm,
            timeout_seconds=_chunk_timeout,
        )

        if result["success"]:
            print(
                f"[MARKER] Chunk {i}/{total_chunks} (pages {start}–{end}): "
                f"DONE ✓ ({result['elapsed']:.1f}s)"
            )
            completed_chunk_dirs.append(chunk_dir)
            succeeded += 1
        else:
            label = "TIMEOUT" if result["timed_out"] else "FAILED"
            print(
                f"[MARKER] Chunk {i}/{total_chunks} (pages {start}–{end}): "
                f"{label} ✗ — {result['error']}"
            )
            failed += 1
            failed_ranges.append(f"{start}–{end}")

            # If marker_single itself isn't found, no point continuing
            if "not found" in (result["error"] or ""):
                raise RuntimeError(result["error"])

    total_elapsed = time.time() - total_start

    # ── Check if we have any usable output ───────────────────────────
    if not completed_chunk_dirs:
        raise RuntimeError(
            f"ALL {total_chunks} chunks failed for '{pdf_path_obj.name}'. "
            f"No usable output. Failed ranges: {failed_ranges}"
        )

    # ── Merge chunk outputs into one combined JSON ───────────────────
    # Sort by directory name to ensure page order
    completed_chunk_dirs.sort(key=lambda d: d.name)

    merged_dir = output_dir_obj / "merged"
    merged_result = _merge_chunk_outputs(
        completed_chunk_dirs, merged_dir, pdf_stem
    )

    # ── Summary ──────────────────────────────────────────────────────
    print(
        f"\n[MARKER] Summary for '{pdf_path_obj.name}':\n"
        f"  Total chunks  : {total_chunks}\n"
        f"  Succeeded     : {succeeded} (newly processed)\n"
        f"  Cached/skipped: {skipped} (from previous run)\n"
        f"  Failed        : {failed}"
        + (f" (pages: {', '.join(failed_ranges)})" if failed_ranges else "")
        + f"\n  Total time    : {total_elapsed:.1f}s\n"
        f"  Merged JSON   : {merged_result['json_path']}\n"
        f"  Merged images : {merged_result['images_dir']}"
    )

    return {
        "json_path": merged_result["json_path"],
        "images_dir": merged_result["images_dir"],
        "pdf_path": pdf_path,
        "chunks_total": total_chunks,
        "chunks_succeeded": succeeded,
        "chunks_failed": failed,
        "chunks_skipped": skipped,
        "total_elapsed": total_elapsed,
    }


# ═══════════════════════════════════════════════════════════════════════════
# CLI entry-point — useful for standalone testing
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(
            "Usage: python -m preprocessing.marker_ocr <input.pdf> <output_dir>\n"
            "\n"
            "Runs marker_single against a PDF (chunked) and prints output paths.\n"
            "Use this to verify Marker is installed correctly and to inspect\n"
            "the JSON output structure before running parse_marker_output.py.\n"
            "\n"
            "Optional flags:\n"
            "  --force-ocr      Force full re-OCR on every page\n"
            "  --chunk-size N   Pages per chunk (default: from settings)\n"
        )
        sys.exit(1)

    _pdf = sys.argv[1]
    _out = sys.argv[2]
    _force = "--force-ocr" in sys.argv
    _chunk = None
    for i, arg in enumerate(sys.argv):
        if arg == "--chunk-size" and i + 1 < len(sys.argv):
            _chunk = int(sys.argv[i + 1])

    result = run_marker(_pdf, _out, force_ocr=_force, chunk_size=_chunk)

    print(f"\n{'─' * 60}")
    print(f"  Marker Output (Chunked)")
    print(f"{'─' * 60}")
    print(json.dumps(result, indent=2, default=str))
    print(f"{'─' * 60}")
