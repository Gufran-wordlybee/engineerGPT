# Engineering RAG Assistant

A vectorless, structure-based RAG system for studying engineering textbooks. Uses TOC-aware section routing instead of vector search — because textbook structure (chapters, sections, headings) is more reliable for topic-matching than embedding similarity.

## Why Vectorless?

- Textbooks already have reliable structure (TOC, chapters, sections)
- Vector chunking breaks equations, derivations, and figure+caption pairs
- At curriculum scale (~10 books), vector DB adds overhead without accuracy benefit
- Structure-based routing mirrors the manual "Cmd+F → read the section" workflow

---

## Architecture Overview

The preprocessing pipeline has **two parallel paths** that converge on a single output format:

```
                          ┌─────────────────────────────────────────┐
                          │              INPUT PDF                   │
                          │         books/raw/<book>.pdf             │
                          └──────────────┬──────────────────────────┘
                                         │
                              ┌──────────▼──────────┐
                              │  scanned_detector   │
                              │  classify_pdf()     │
                              └──────────┬──────────┘
                                         │
                          ┌──────────────┴──────────────┐
                          │                             │
                   classification:                classification:
                     "text"                          "scanned"
                          │                             │
              ┌───────────▼──────────┐      ┌───────────▼───────────┐
              │   TEXT-BASED PATH    │      │   SCANNED PATH        │
              │   (pymupdf)          │      │   (Marker OCR)        │
              │                      │      │                       │
              │  1. extract_toc      │      │  1. marker_ocr        │
              │  2. split_sections   │      │     run_marker()      │
              │  3. extract_images   │      │     (CHUNKED)         │
              │                      │      │  2. parse_marker_output│
              └───────────┬──────────┘      │     build_sections()  │
                          │                 └───────────┬───────────┘
                          │                             │
                          └──────────────┬──────────────┘
                                         │
                              ┌──────────▼──────────┐
                              │   build_index.py    │
                              │   (UNCHANGED)       │
                              └──────────┬──────────┘
                                         │
                          ┌──────────────▼──────────────────────┐
                          │           OUTPUT                     │
                          │  books/processed/<book>/             │
                          │  ├── sections/<section_id>.json      │
                          │  ├── images/*.png                    │
                          │  └── index.json                      │
                          └─────────────────────────────────────┘
```

**Key design principle**: Both paths produce the exact same output contract (`sections/*.json` with identical field names and structure), so everything downstream — `build_index.py`, `core/router.py`, `core/generator.py` — requires **zero changes** regardless of which path processed a book.

---

## Scanned-PDF OCR: How the Performance Fixes Work

Marker (`marker-pdf`) on a 16GB M3 MacBook (no CUDA GPU) was taking **3+ hours** on 600+ page scanned books — to the point where runs had to be killed. Three compounding causes were identified and fixed:

### Problem 1: Force-OCR on every page (fixed)

**Before:** `MARKER_FORCE_OCR = True` by default — every page re-OCR'd from scratch, even pages with a usable embedded text layer.

**After:** Two-threshold smart OCR decision:
- **avg < 5 chars/page** → `force_ocr = True` (pure image scan, no text at all)
- **avg 5–25 chars/page** → `force_ocr = False` (partial text layer exists — Marker uses it where it can, 30–70% faster)
- **avg ≥ 25 chars/page** → pymupdf text path (no OCR needed)

This is automatic: `scanned_detector.py` classifies the book, passes `needs_force_ocr` to `marker_ocr.py`, which uses it to decide per-book.

### Problem 2: No chunking (fixed)

**Before:** One `marker_single` call for the entire 600-page PDF, timeout set to `None`. A crash or Cmd+C loses **all progress** — hours of work gone.

**After:** The PDF is split into chunks of 50 pages (configurable). Each chunk:
- Has its own **timeout** (default 900s / 15 min)
- Has its own output directory
- Is **cached on completion** — re-running after a crash resumes from where it left off
- On timeout: logged and skipped, the rest of the book continues

**Worst case on crash:** lose ~5–10 minutes (one chunk), not 3+ hours.

### Problem 3: No GPU (solved via Google Colab)

**Before:** Marker's vision-transformer OCR on CPU is 10–20x slower than GPU.

**After:** A ready-to-use **Google Colab notebook** (`colab/marker_colab_runner.ipynb`) runs Marker on a **free T4 GPU**. Expected speedup: 600 pages in ~15–35 minutes instead of 3+ hours.

---

## Folder Structure

```
Phase-0/
├── books/
│   ├── raw/                        ← Drop PDFs here (input)
│   └── processed/                  ← Pipeline output (per book)
│       └── <book_name>/
│           ├── sections/           ← Per-section JSON files
│           │   ├── chapter-1.json
│           │   ├── 1-1-introduction.json
│           │   └── ...
│           ├── images/             ← Extracted images & equations
│           │   ├── chapter-1_img_1.png
│           │   └── ...
│           ├── index.json          ← Hierarchical search index
│           ├── qa_report.txt       ← Section length outlier report
│           ├── confusable_pairs.json ← High-overlap section pairs
│           └── marker_raw/         ← Raw Marker output (scanned only)
│               ├── chunks/         ← Per-chunk output dirs
│               │   ├── chunk_0000_0049/
│               │   ├── chunk_0050_0099/
│               │   └── ...
│               └── merged/         ← Combined output
│                   ├── <book>.json
│                   └── images/
│
├── preprocessing/                  ← Runs once per book
│   ├── __init__.py
│   ├── run_pipeline.py             ← Main orchestrator (the fork point)
│   ├── scanned_detector.py         ← Classifies PDF as scanned/text
│   ├── marker_ocr.py               ← Runs Marker CLI (chunked, with resume)
│   ├── parse_marker_output.py      ← Converts Marker JSON → sections
│   ├── extract_toc.py              ← Text path: TOC extraction
│   ├── split_sections.py           ← Text path: section splitting
│   ├── extract_images.py           ← Text path: image extraction
│   └── build_index.py              ← Shared: builds index.json
│
├── core/                           ← Runs per query (unchanged)
│   ├── router.py
│   ├── generator.py
│   └── pipeline.py
│
├── config/
│   └── settings.py                 ← All configurable thresholds & paths
│
├── colab/
│   └── marker_colab_runner.ipynb   ← Google Colab notebook for GPU OCR
│
└── interfaces/                     ← CLI and web entrypoints
```

---

## Setup

```bash
# Clone and enter
git clone <repo-url>
cd engineering-rag-assistant

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env .env.local  # Edit with your API keys
```

### Important Notes on `marker-pdf`

- **Python 3.10+ required** (confirm with `python --version`)
- Pulls in heavy dependencies: `torch`, `surya-ocr`, `transformers`, etc.
- **First run downloads model weights** (~1-2 GB). This is a one-time operation.
- Runs on CPU if no GPU is available — slower but functional. Use the Colab notebook for heavy books.
- If you only process text-based PDFs, `marker-pdf` is imported but never called — no overhead.

---

## Usage

### Quick Start: Process a Book Locally

```bash
# Auto-detects scanned vs text, processes accordingly
python -m preprocessing.run_pipeline books/raw/fluid_mechanics.pdf

# Process all books in books/raw/
python -m preprocessing.run_pipeline

# Re-process even if output already exists
python -m preprocessing.run_pipeline --force
```

### Override Auto-Detection

```bash
# Force the scanned/Marker OCR path
python -m preprocessing.run_pipeline books/raw/old_scan.pdf --force-scanned

# Force the text-based/pymupdf path
python -m preprocessing.run_pipeline books/raw/modern_book.pdf --force-text
```

### Standalone Module Testing

Each module can be tested independently:

```bash
# Test PDF classification only (useful for threshold tuning)
python -m preprocessing.scanned_detector books/raw/your_book.pdf

# Test TOC extraction only
python -m preprocessing.extract_toc books/raw/your_book.pdf

# Test Marker OCR only (chunked, writes raw output to a directory)
python -m preprocessing.marker_ocr books/raw/scanned_book.pdf output/marker_test

# Test Marker output parsing only
python -m preprocessing.parse_marker_output <marker.json> <images_dir> <output_dir>
```

### Marker OCR CLI Options

```bash
# Force full re-OCR on every page (for pure image-only scans)
python -m preprocessing.marker_ocr books/raw/book.pdf output/ --force-ocr

# Use a custom chunk size (e.g. 30 pages per chunk)
python -m preprocessing.marker_ocr books/raw/book.pdf output/ --chunk-size 30
```

### Phase 1 Router Evaluation

```bash
# Check hand-labeled questions without making API calls
python -m evaluation.eval_router --all --validate-only

# Run live router accuracy evaluation
python -m evaluation.eval_router --all

# Evaluate a single book with detailed misses
python -m evaluation.eval_router --book ai --verbose
```

The hard Phase 1 gate is top-1 accuracy of at least 90% on each book's hand-written question set. Validation-only mode is useful before spending API calls: it confirms every `expected_sections` label exists in the flattened, routable index.

---

## Using Google Colab for Heavy Scanned Books (Recommended)

For 600+ page scanned books, running Marker on your Mac's CPU is impractical. Use the included Colab notebook to run on a **free T4 GPU** instead.

### When to Use Colab

| Situation | Recommendation |
|---|---|
| Text-based PDF (good embedded text) | Run locally — `pymupdf` path, fast |
| Small scanned PDF (< 100 pages) | Run locally — Marker on CPU is fine |
| Large scanned PDF (100+ pages) | **Use Colab** — 10–20x faster |
| Colab GPU unavailable | Run locally with chunking — slower but crash-safe |

### Step-by-Step Colab Workflow

1. **Upload your PDF to Google Drive**
   ```
   Google Drive → engineering-rag-assistant/books/raw/<book_name>.pdf
   ```

2. **Open the notebook in Colab**
   - Upload `colab/marker_colab_runner.ipynb` to Colab (or open from Drive)
   - **Set GPU runtime**: `Runtime` → `Change runtime type` → **T4 GPU**

3. **Edit Cell 5** — set `BOOK_NAME` to match your PDF filename (without `.pdf`)

4. **Run all cells** — the notebook will:
   - Install `marker-pdf`
   - Mount your Google Drive
   - Process the PDF in 100-page chunks (with resume support)
   - Merge outputs into a single JSON + images directory
   - Save everything to `Google Drive/engineering-rag-assistant/marker_output/<book_name>/merged/`

5. **Get the output back to your Mac** (pick one):
   - **Google Drive for Desktop** (recommended): output appears as a local folder automatically
   - **Manual download**: download the zip from Colab's file browser

6. **Run the local post-processing**
   ```bash
   # Parse Marker's output into sections (fast, pure Python)
   python -m preprocessing.parse_marker_output \
       <path/to/merged.json> \
       <path/to/merged/images> \
       books/processed/<book_name> \
       "Book Title"

       python -m preprocessing.parse_marker_output \
    books/processed/coa/marker_raw/merged/coa.json \
    books/processed/coa/marker_raw/merged/images \
    books/processed/coa \
    "coa"

   # Build the search index
   python -m preprocessing.build_index "Book Title" books/processed/<book_name>
   ```

### Session Disconnects & Resume

Colab free-tier sessions can disconnect (idle timeout, 12-hour cap, demand preemption). The chunking ensures this doesn't lose progress:

- Each completed chunk is saved to Google Drive immediately
- Re-running Cell 6 **skips already-completed chunks** and resumes from where it left off
- Worst case: you lose the single chunk that was in progress (~5–15 minutes of work)

### Kaggle Notebooks as Fallback

If Colab's GPU queue is unavailable (free-tier availability isn't guaranteed), use the same notebook on **Kaggle Notebooks**:

- Kaggle offers comparable free GPU access (P100/T4, ~30 hours/week)
- Often more predictable session availability than Colab
- Supports background execution (closed browser tab doesn't kill the run)
- Only difference: use Kaggle's dataset upload instead of Google Drive for input/output
- Same notebook cells work on both platforms — no code changes needed

---

## Configuration Reference

All settings live in `config/settings.py` and can be overridden via environment variables in `.env`.

### Scanned-PDF Detection

| Setting | Default | Purpose |
|---|---|---|
| `SCANNED_DETECTION_MIN_CHARS_PER_PAGE` | `25` | Below this avg chars/page → classified as "scanned" |
| `FORCE_OCR_MAX_CHARS_PER_PAGE` | `5` | Below this avg → force full re-OCR; between this and MIN_CHARS → let Marker decide per-page |

### Marker OCR

| Setting | Default | Purpose |
|---|---|---|
| `MARKER_CLI_BINARY` | `marker_single` | Path to the Marker CLI binary |
| `MARKER_OUTPUT_FORMAT` | `json` | Output format (must be `json` for parser) |
| `MARKER_FORCE_OCR` | `False` | Global override — force re-OCR on every scanned book. Usually leave `False` and let the smart detection decide. |
| `MARKER_USE_LLM` | `False` | Enable LLM-assisted extraction (better tables/math, adds cost) |
| `MARKER_CHUNK_SIZE_PAGES` | `50` | Pages per chunk for local CPU processing |
| `MARKER_CHUNK_TIMEOUT_SECONDS` | `900` | Timeout per chunk in seconds (15 min default) |

### Tuning Tips

- **`FORCE_OCR_MAX_CHARS_PER_PAGE`**: Test against your actual scanned books. Run `python -m preprocessing.scanned_detector <book.pdf>` and check the `per_page_chars` output. If most pages have 0 chars, the default of 5 is fine. If your scans have garbled-but-present text (common with university-distributed scans), you might raise this to 10–15.

- **`MARKER_CHUNK_SIZE_PAGES`**: On CPU, 50 is safe. On GPU (Colab), use 100–200 for fewer subprocess calls. Smaller chunks = more resume points but more overhead.

- **`MARKER_CHUNK_TIMEOUT_SECONDS`**: 900s (15 min) for 50 pages on CPU gives generous headroom. On GPU, you can reduce this since processing is much faster.

---

## How Each Module Works

### `scanned_detector.py` — The Fork Point

**Question it answers:** "Does this PDF have extractable text, or is it just images?"

**How:**
1. Opens the PDF with pymupdf
2. Samples a spread of pages (first 5, middle 5, last 5)
3. Calls `page.get_text("text")` on each and counts characters
4. Computes **average characters per page**
5. Two-threshold classification:
   - `avg < 5` → scanned, `needs_force_ocr = True` (pure image scan)
   - `avg 5–25` → scanned, `needs_force_ocr = False` (partial text layer)
   - `avg ≥ 25` → text (pymupdf path)

**Output:** A result dict with classification, `needs_force_ocr`, stats, and per-page char counts.

---

### `marker_ocr.py` — The OCR Engine Wrapper (Chunked)

**What it does:** Runs the `marker_single` CLI against a scanned PDF, in **chunks** with resume support.

**Why chunked?** A single `marker_single` call on 600 pages means losing all progress on crash/Cmd+C. Chunking turns "lose 3 hours" into "lose at most 5–10 minutes."

**Chunking flow:**
1. Get page count via pymupdf (cheap, just reads metadata)
2. Split into chunks of `MARKER_CHUNK_SIZE_PAGES` (default 50)
3. For each chunk:
   - If already completed (has `.json` output) → skip (resume support)
   - Call `marker_single --page_range "start-end"` with per-chunk timeout
   - On timeout: log warning, skip to next chunk
4. Merge all chunk outputs into one combined JSON

**Key flags:**
- `--force_ocr` — re-OCR even if there's a partial text layer (auto-decided by smart detection)
- `--use_llm` — opt-in LLM-assisted extraction (better tables/math, but adds cost)
- `--output_format json` — structured JSON block tree (not markdown)
- `--page_range "0-49"` — process only this page range (enables chunking)

**Why CLI instead of Python API?** There's a known bug (marker issue #906) where the Python API's `output_format` config for `json`/`chunks` is silently ignored. The CLI works correctly.

---

### `parse_marker_output.py` — The Bridge

**The most important module for understanding the architecture.** This is what makes the two paths converge.

**What it does:**
1. Loads Marker's JSON block tree
2. Flattens the hierarchical blocks into document reading order
3. Walks blocks with a state machine:
   - `SectionHeader` → start new section
   - `Text` → append to current section's text
   - `Picture`/`Figure` → save image, add to `images` list
   - `Equation` → save image, add to `equations` list (major win!)
   - `Table` → render as text in section body
4. Generates `section_id` using the same `_slugify()` as `split_sections.py`
5. Builds parent/child hierarchy
6. Writes each section to `sections/<id>.json` with the **exact same schema**

**Why equations are a big deal:** The existing pymupdf path (`extract_images.py`) cannot reliably distinguish equations from regular figures — it uses font analysis and regex heuristics. Marker's layout model natively identifies equations as a distinct block type.

---

### `run_pipeline.py` — The Orchestrator

The main entry point. Here's what happens when you process a book:

1. **Derive book name** from PDF filename
2. **Check if already processed** (skip unless `--force`)
3. **Classify the PDF** via `scanned_detector.classify_pdf()`
4. **Fork:**
   - `"text"` → `extract_toc()` → `split_sections()` → `extract_images()`
   - `"scanned"` → `run_marker()` → `build_sections()` (with smart force-OCR from detection result)
5. **Converge:** `build_index()` runs on whichever path's output
6. **Log summary** with path taken, section count, timing

---

## Output Contract

Every section JSON (from either path) has this exact shape:

```json
{
  "section_id": "3-2-fluid-dynamics",
  "title": "3.2 Fluid Dynamics",
  "level": 2,
  "chapter": "Chapter 3: Fundamentals",
  "parent_id": "chapter-3-fundamentals",
  "children": ["3-2-1-bernoulli", "3-2-2-continuity"],
  "start_page": 42,
  "end_page": 55,
  "text": "Full section body text...",
  "images": [
    {"path": "images/3-2-fluid-dynamics_img_1.png", "type": "figure", "caption": "Figure 3.1"}
  ],
  "equations": [
    {"path": "images/3-2-fluid-dynamics_eq_1.png", "type": "equation", "label": "3.14"}
  ]
}
```

This is what `build_index.py` reads. As long as the field names match, it works — it doesn't care which path produced the file.

---

## Phase Roadmap

| Phase | Status | Description |
|-------|--------|-------------|
| **0 — Preprocessing** | ✅ Built | TOC extraction, section splitting, image extraction, index building |
| **0.5 — Scanned OCR** | ✅ Built | Marker-based OCR path for scanned PDFs (chunked, smart OCR, Colab GPU support) |
| **1 — Router** | ✅ Built, needs live accuracy eval | LLM-based section selection from index.json |
| **2 — Generator** | ✅ Built, needs live API validation | Grounded answers with optional visuals |
| **3 — CLI** | 🔲 Planned | Interactive command-line question/answer loop |
| **4 — Weekly Updates** | 🔲 Planned | Drop-in book addition workflow |
| **5 — Streamlit** | 🔲 Planned | Web UI for browser-based access |

## Phase 2: Answer Generation

`core.pipeline.run_query(book_name, question)` is the single application API.
It routes with `GROQ_LLM_MODEL_NAME` (for example `openai/gpt-oss-120b`), then
answers from the selected section JSON with `GENERATE_LLM_MODEL`.

The generator uses Groq's OpenAI-compatible endpoint, so no second SDK is
needed. In `Phase-0/.env`, set `GENERATE_LLM_MODEL` to the Qwen model you want
and either set `GENERATE_LLM_Model_API` or leave it blank to reuse
`GROQ_LLM_API_KEY`. Visuals are attached only from the routed sections. Qwen
3.6 permits three images, but this project defaults to one image and 12K source
characters to fit the current 8K TPM account limit; raise the `.env` limits
only after confirming your Groq tier can accommodate the larger request.

```bash
cd Phase-0
python -m core.generator  # local image-gate self-check
python -c 'from core.pipeline import run_query; print(run_query("coa", "What is a multiplexer used for")["answer"])'
```

---

## Validation Checklist

After setting up, verify the pipeline works correctly:

- [ ] Run `scanned_detector.py` against every book in your library and manually confirm each classification
- [ ] Confirm the `needs_force_ocr` flag matches your expectation for each book
- [ ] Run Marker locally on a small scanned PDF (< 100 pages) to verify chunking works
- [ ] Run the Colab notebook on a large scanned PDF and confirm output matches local format
- [ ] Verify resume works: kill a chunked run midway, re-run, confirm it skips completed chunks
- [ ] Confirm every `Figure`/`Picture` block has a corresponding image saved under `images/`
- [ ] Confirm every `Equation` block lands in `equations`, not `images`
- [ ] Run `build_index.py` (unmodified) against Marker output and confirm `index.json` matches the structure of a text-based book
- [ ] Run `python -m evaluation.eval_router --all --validate-only` to confirm every hand-labeled router question points to a real routable section
- [ ] Run `python -m evaluation.eval_router --all` with network/API access and confirm top-1 router accuracy is at least 90%

---

## Troubleshooting

### "marker_single not found"

Marker isn't installed or isn't on your PATH:
```bash
pip install marker-pdf
# Or set the binary path in .env:
MARKER_CLI_BINARY=/path/to/marker_single
```

### "Chunk timed out"

A single chunk took longer than `MARKER_CHUNK_TIMEOUT_SECONDS`. The chunk is skipped and the rest of the book continues. Options:
- Increase `MARKER_CHUNK_TIMEOUT_SECONDS` in `.env` (e.g. `1800` for 30 minutes)
- Decrease `MARKER_CHUNK_SIZE_PAGES` (fewer pages per chunk = less per-chunk time)
- Use the Colab notebook for GPU acceleration

### "ALL chunks failed"

If every single chunk fails, the pipeline raises a `RuntimeError`. Common causes:
- `marker_single` not installed (check with `which marker_single`)
- Corrupted PDF (try opening in a PDF viewer)
- Out of memory (reduce `MARKER_CHUNK_SIZE_PAGES`)

### Marker is too slow on CPU

This is expected for large books. Marker's OCR uses vision-transformer models that are 10–20x slower on CPU vs GPU. Solutions:
- **Use the Colab notebook** (recommended for 100+ page books)
- Reduce `MARKER_CHUNK_SIZE_PAGES` to get partial results faster
- Make sure `MARKER_FORCE_OCR` isn't set to `True` in `.env` — let the smart detection decide

### "No TOC found for text-based PDF"

The text-based path requires detectable headings (via bookmarks, font size, or regex). If none are found, the book is skipped. Options:
- Force the Marker path: `--force-scanned` (Marker's layout model detects headers independently)
- Manually add bookmarks to the PDF using a PDF editor

### "Classification seems wrong"

Adjust the thresholds:
```bash
# In .env
SCANNED_DETECTION_MIN_CHARS_PER_PAGE=50  # increase to be more aggressive about OCR
FORCE_OCR_MAX_CHARS_PER_PAGE=10          # increase if partial-text scans still need full OCR
```

Or override per-book: `--force-scanned` / `--force-text`

### Colab session disconnected mid-run

This is normal for Colab free tier. Just re-run Cell 6 — it skips all completed chunks and resumes from where it left off. Your completed work is safe in Google Drive.

---

## Requirements

- Python 3.10+
- PyMuPDF (for PDF processing — both paths)
- python-dotenv (for environment management)
- marker-pdf (for scanned-PDF OCR — only called when a PDF is classified as scanned)
