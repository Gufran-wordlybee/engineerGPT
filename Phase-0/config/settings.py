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
LLM_API_KEY: str = os.getenv("LLM_API_KEY", "")
LLM_MODEL_NAME: str = os.getenv("LLM_MODEL_NAME", "")
VISION_MODEL_NAME: str = os.getenv("VISION_MODEL_NAME", "")

# Provider selection — auto-detected from model name if not set.
# Valid values: "gemini", "groq", "openai", "anthropic"
# Gemini and Groq have FREE tiers — recommended for getting started.
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "")

# Custom base URL for OpenAI-compatible providers (e.g. local Ollama).
# Leave empty to use the default URL for the detected provider.
LLM_BASE_URL: str = os.getenv("LLM_BASE_URL", "")

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
# Requires a valid LLM_API_KEY.  Falls back to TF-IDF-only if disabled
# or if no API key is configured.
LLM_ABSTRACTS_ENABLED: bool = os.getenv("LLM_ABSTRACTS_ENABLED", "false").lower() in ("true", "1", "yes")

# ---------------------------------------------------------------------------
# Router settings (Phase 1)
# ---------------------------------------------------------------------------
# Model name for routing calls.  Defaults to the same model used for
# abstracts, but can be overridden to use a cheaper/faster model for
# routing while keeping a stronger model for generation.
ROUTER_MODEL_NAME: str = os.getenv("ROUTER_MODEL_NAME", "") or LLM_MODEL_NAME

# Deterministic routing — temperature=0 avoids creative variation.
ROUTER_TEMPERATURE: float = float(os.getenv("ROUTER_TEMPERATURE", "0"))

# Books with more than this many sections use two-stage routing
# (chapter pick → section pick).  Smaller books use a single flat prompt.
TWO_STAGE_THRESHOLD: int = int(os.getenv("TWO_STAGE_THRESHOLD", "30"))

# Default number of sections returned by the router.
ROUTER_TOP_K_DEFAULT: int = int(os.getenv("ROUTER_TOP_K_DEFAULT", "2"))

