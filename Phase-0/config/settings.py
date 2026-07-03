"""Configuration settings for the Engineering Book RAG Assistant.

Loads environment variables from .env and defines constants for
TOC extraction, heading detection, and image processing.
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
# Image extraction constants
# ---------------------------------------------------------------------------
# Minimum width or height (pixels) to keep an extracted image.
MIN_IMAGE_SIZE: int = 50

# ---------------------------------------------------------------------------
# Keyword extraction
# ---------------------------------------------------------------------------
TOP_N_KEYWORDS: int = 15

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
