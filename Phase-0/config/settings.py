"""Configuration settings for the Engineering Book RAG Assistant.

Loads environment variables from .env and defines constants for
TOC extraction, heading detection, section splitting, image/equation
extraction, and index building.
"""

import os
import re
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load .env from project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# ---------------------------------------------------------------------------
# LLM / API settings
# ---------------------------------------------------------------------------
GEMENI_LLM_API_KEY: str = os.getenv("GEMENI_LLM_API_KEY", "")
GROQ_LLM_API_KEY: str = os.getenv("GROQ_LLM_API_KEY", "")
GEMENI_LLM_MODEL_NAME: str = os.getenv("GEMENI_LLM_MODEL_NAME", "")
GROQ_LLM_MODEL_NAME: str = os.getenv("GROQ_LLM_MODEL_NAME", "")

# Phase 2 uses the same Groq OpenAI-compatible API as the router. Keep a
# separate model/key so routing can stay on OSS-120B while answers use Qwen.
# The mixed-case API variable is supported because it is the name used in the
# project .env; the uppercase alias is accepted for shell-friendly deployments.
GENERATE_LLM_MODEL: str = os.getenv(
    "GENERATE_LLM_MODEL",
    os.getenv("GENERATOR_LLM_MODEL_NAME", "qwen/qwen3.6-27b"),
)
GENERATE_LLM_MODEL_API: str = (
    os.getenv("GENERATE_LLM_Model_API")
    or os.getenv("GENERATE_LLM_MODEL_API")
    or GROQ_LLM_API_KEY
)
GENERATOR_LLM_TIMEOUT_SECONDS: float = float(
    os.getenv("GENERATOR_LLM_TIMEOUT_SECONDS", "45")
)
GENERATOR_MAX_TOKENS: int = int(os.getenv("GENERATOR_MAX_TOKENS", "2000"))
GENERATOR_TEMPERATURE: float = float(os.getenv("GENERATOR_TEMPERATURE", "0.3"))
GENERATOR_MAX_CONTEXT_CHARS: int = int(
    os.getenv("GENERATOR_MAX_CONTEXT_CHARS", "12000")
)
VISION_MODEL_NAME: str = os.getenv("VISION_MODEL_NAME", GENERATE_LLM_MODEL)
VISION_MAX_IMAGES_PER_REQUEST: int = int(
    os.getenv("VISION_MAX_IMAGES_PER_REQUEST", "1")
)
VISION_MAX_IMAGE_SIZE_MB: int = int(os.getenv("VISION_MAX_IMAGE_SIZE_MB", "20"))

# Router-specific settings. These reuse the Groq/OpenAI-compatible setup used
# by index abstract generation, so Phase 1 does not introduce another provider.
ROUTER_TOP_K: int = int(os.getenv("ROUTER_TOP_K", "3"))
ROUTER_SHORTLIST_N: int = int(os.getenv("ROUTER_SHORTLIST_N", "20"))
ROUTER_LLM_MODEL_NAME: str = os.getenv(
    "ROUTER_LLM_MODEL_NAME",
    GROQ_LLM_MODEL_NAME,
)
ROUTER_LLM_TIMEOUT_SECONDS: float = float(
    os.getenv("ROUTER_LLM_TIMEOUT_SECONDS", "20")
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BOOKS_RAW_PATH: Path = Path(
    os.getenv("BOOKS_RAW_PATH", PROJECT_ROOT / "books" / "raw")
).resolve()
BOOKS_PROCESSED_PATH: Path = Path(
    os.getenv("BOOKS_PROCESSED_PATH", PROJECT_ROOT / "books" / "processed")
).resolve()

# ---------------------------------------------------------------------------
# TOC extraction constants
# ---------------------------------------------------------------------------
# A heading must be at least this ratio × body-text font-size to qualify.
MIN_HEADING_FONT_RATIO: float = 1.2

# Font-size ratio thresholds for heading levels
HEADING_FONT_RATIO_L1: float = 1.6   # Level 1 heading threshold
HEADING_FONT_RATIO_L2: float = 1.3   # Level 2 heading threshold

# Minimum number of PDF bookmarks before we trust them as a TOC source.
MIN_BOOKMARK_ENTRIES: int = 3

# ---------------------------------------------------------------------------
# Heading regex patterns  →  heading level
# Ordered most-specific first: sub-subsection → subsection → chapter/part
# so that e.g. "3.2.1 ..." isn't caught by the "\d+\.\d+" pattern.
# ---------------------------------------------------------------------------
HEADING_REGEX_PATTERNS: list[tuple[re.Pattern, int]] = [
    (re.compile(r"^\d+\.\d+\.\d+\s+"),   3),   # e.g. "3.2.1 Bernoulli's …"
    (re.compile(r"^\d+\.\d+\s+"),         2),   # e.g. "3.2 Fluid Dynamics"
    (re.compile(r"^Chapter\s+\d+", re.IGNORECASE), 1),
    (re.compile(r"^Part\s+\d+",   re.IGNORECASE), 1),
    (re.compile(r"^\d+\s+[A-Z]"),         1),   # chapter number + title
]

# ---------------------------------------------------------------------------
# Section splitting constants
# ---------------------------------------------------------------------------
# Sections with fewer characters than this are auto-merged into the next
# sibling (or folded into the parent chapter).
MIN_SECTION_CHARS: int = 200

# How many pages to sample from the start of a page to detect repeating
# headers/footers.  We check the first and last N lines of each page.
HEADER_FOOTER_LINES: int = 3

# Minimum fraction of pages a line must appear on to be classified as a
# repeating header/footer and stripped from the extracted text.
HEADER_FOOTER_MIN_RATIO: float = 0.4

# Section length outlier thresholds (ratio to median section length).
# Sections outside this range are flagged in the QA report.
SECTION_OUTLIER_LONG: float = 3.0    # > 3× median → suspiciously long
SECTION_OUTLIER_SHORT: float = 0.15  # < 0.15× median → suspiciously short

# ---------------------------------------------------------------------------
# Image & visual-region extraction constants
# ---------------------------------------------------------------------------
# Minimum width or height (pixels) to keep an extracted raster image.
MIN_IMAGE_SIZE: int = 50

# DPI for rasterizing vector diagrams and equation regions.
RASTERIZE_DPI: int = 200

# Minimum number of vector drawing paths in a cluster to qualify as a diagram.
DIAGRAM_MIN_PATHS: int = 5

# Maximum gap (in PDF points) between drawing paths to merge them into the
# same cluster.  72 pt = 1 inch.
DIAGRAM_CLUSTER_GAP: float = 30.0

# Minimum area (in sq PDF points) for a diagram region to be kept.
DIAGRAM_MIN_AREA: float = 5000.0

# ---------------------------------------------------------------------------
# Equation detection
# ---------------------------------------------------------------------------
# Regex for numbered display equations like "(3.14)" at end of line.
EQUATION_NUMBER_RE: re.Pattern = re.compile(
    r"\((\d+[\.\-]\d+(?:[\.\-]\d+)?)\)\s*$"
)

# Regex for figure/diagram captions: "Figure 2.1", "Fig. 2.1", "FIGURE 2.1"
FIGURE_CAPTION_RE: re.Pattern = re.compile(
    r"(?:Figure|Fig\.?|FIGURE|FIG\.?)\s*(\d+[\.\-]\d+(?:[a-z])?)",
    re.IGNORECASE,
)

# Unicode code-point ranges for common math symbols.
# Used to detect lines with high math-symbol density.
MATH_SYMBOL_CHARS: set[str] = set(
    "∀∃∄∅∆∇∈∉∊∋∌∍∎∏∐∑−∓∔∕∖∗∘∙√∛∜∝∞∟∠∡∢∣∤∥∦∧∨∩∪∫∬∭∮∯∰∱∲∳"
    "∴∵∶∷∸∹∺∻∼∽∾∿≀≁≂≃≄≅≆≇≈≉≊≋≌≍≎≏≐≑≒≓≔≕≖≗≘≙≚≛≜≝≞≟≠≡≢≣≤≥≦≧≨≩"
    "≪≫≬≭≮≯≰≱≲≳≴≵≶≷≸≹≺≻≼≽≾≿⊀⊁⊂⊃⊄⊅⊆⊇⊈⊉⊊⊋⊌⊍⊎⊏⊐⊑⊒⊓⊔⊕⊖⊗⊘⊙"
    "αβγδεζηθικλμνξοπρστυφχψωΓΔΘΛΞΠΣΦΨΩ"
    "×÷±≈≠≤≥→←↑↓↔⇒⇐⇔∂∇"
)

# Math font name fragments (case-insensitive search in span font name).
MATH_FONT_FRAGMENTS: list[str] = [
    "symbol", "math", "cmmi", "cmsy", "cmex", "cmr",
    "msam", "msbm", "eufm", "rsfs", "stix",
]

# ---------------------------------------------------------------------------
# Keyword / index extraction
# ---------------------------------------------------------------------------
TOP_N_KEYWORDS: int = 15

# Whether to generate LLM-based section abstracts at index-build time.
# Requires a configured Groq/Gemini-compatible LLM key. Falls back to
# TF-IDF-only if disabled or if no API key is configured.
LLM_ABSTRACTS_ENABLED: bool = os.getenv("LLM_ABSTRACTS_ENABLED", "false").lower() in ("true", "1", "yes")

# ---------------------------------------------------------------------------
# Scanned-PDF detection
# ---------------------------------------------------------------------------
# If the average number of extractable characters per sampled page is
# below this threshold, the PDF is classified as "scanned" and routed
# through the Marker OCR path instead of the pymupdf text-extraction path.
#
# Typical values:
#   - Pure scanned PDFs (image-only): ~0 chars/page
#   - Partial-OCR scans (garbled text layer): 5–20 chars/page
#   - Text-based PDFs: hundreds to thousands of chars/page
#   - Tune this after testing against your actual book library.
SCANNED_DETECTION_MIN_CHARS_PER_PAGE: int = int(
    os.getenv("SCANNED_DETECTION_MIN_CHARS_PER_PAGE", "25")
)

# If avg chars/page is below THIS threshold, we consider the PDF a "true
# image-only scan" with no usable text layer at all.  Marker should be told
# to force a full re-OCR (--force_ocr) in this case.
#
# If avg chars/page is BETWEEN this and SCANNED_DETECTION_MIN_CHARS_PER_PAGE,
# the PDF has a partial/garbled text layer.  Marker can try to use whatever
# text layer exists first, which is significantly faster than re-OCR'ing
# every page from scratch.
#
# Example with defaults (FORCE_OCR=5, MIN_CHARS=25):
#   avg 0–5   → force_ocr=True   (pure image scan, no text at all)
#   avg 5–25  → force_ocr=False  (has some text, let Marker decide per-page)
#   avg 25+   → pymupdf path     (good text layer, no OCR needed)
FORCE_OCR_MAX_CHARS_PER_PAGE: int = int(
    os.getenv("FORCE_OCR_MAX_CHARS_PER_PAGE", "5")
)

# ---------------------------------------------------------------------------
# Marker OCR settings (for scanned-PDF preprocessing)
# ---------------------------------------------------------------------------
# Path or name of the Marker CLI binary.  Override via environment variable
# if marker_single isn't on your PATH or has a different name.
MARKER_CLI_BINARY: str = os.getenv("MARKER_CLI_BINARY", "marker_single")

# Output format for Marker.  Must be "json" for our parser to work.
# Don't change this unless Marker's format naming changes in a future version.
MARKER_OUTPUT_FORMAT: str = os.getenv("MARKER_OUTPUT_FORMAT", "json")

# Force a full re-OCR even if the PDF has an existing (possibly garbled)
# text layer.  Now defaults to False — the pipeline decides automatically
# based on FORCE_OCR_MAX_CHARS_PER_PAGE.  Set to True here only if you
# want to unconditionally re-OCR every scanned book regardless.
MARKER_FORCE_OCR: bool = os.getenv("MARKER_FORCE_OCR", "false").lower() in ("true", "1", "yes")

# Enable LLM-assisted extraction (better table merging, inline math).
# Requires GOOGLE_API_KEY (Gemini) or an Ollama backend.
# Opt-in only — adds cost, latency, and an external API dependency.
MARKER_USE_LLM: bool = os.getenv("MARKER_USE_LLM", "false").lower() in ("true", "1", "yes")

# ---------------------------------------------------------------------------
# Marker chunking settings
# ---------------------------------------------------------------------------
# Instead of processing the entire PDF in one subprocess call (which means
# losing ALL progress if you Cmd+C or the process dies), we split the book
# into page-range chunks and process each chunk separately.
#
# Benefits:
#   - A crash/timeout only loses the CURRENT chunk (~50 pages), not the
#     whole 600-page book.
#   - Completed chunks are cached to disk, so re-running automatically
#     resumes from where it left off.
#   - Each chunk has its own timeout, so one stuck chunk doesn't block
#     the rest.
#
# How many pages per chunk.  50 is a good default for CPU.
# On GPU (Colab), you can increase this to 100–200 for fewer subprocess
# calls with minimal risk, since GPU processing is much faster.
MARKER_CHUNK_SIZE_PAGES: int = int(
    os.getenv("MARKER_CHUNK_SIZE_PAGES", "50")
)

# Timeout in seconds PER CHUNK (not per book).
# 50 pages on M3 CPU ≈ 5–10 minutes, so 900s (15 min) gives generous
# headroom.  If a chunk times out, it's logged and skipped — the rest of
# the book continues processing.
MARKER_CHUNK_TIMEOUT_SECONDS: int = int(
    os.getenv("MARKER_CHUNK_TIMEOUT_SECONDS", "900")
)
