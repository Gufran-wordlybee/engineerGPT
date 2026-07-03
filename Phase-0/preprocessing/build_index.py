"""
build_index.py — Generate the flattened TOC index (index.json) for Phase 1's router.

Reads all section JSON files produced by split_sections, extracts TF-based
keywords from each section's text, and writes a single index.json that the
query router can load at startup.
"""

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from config.settings import TOP_N_KEYWORDS

# ---------------------------------------------------------------------------
# Hardcoded English stopwords (~200 common + academic-filler terms)
# ---------------------------------------------------------------------------
STOPWORDS: set[str] = {
    # ---- articles / determiners ----
    "a", "an", "the", "this", "that", "these", "those", "my", "your", "his",
    "her", "its", "our", "their", "some", "any", "each", "every", "all",
    "both", "few", "more", "most", "other", "no", "nor", "not", "only",
    "own", "same", "such", "much", "many",
    # ---- pronouns ----
    "i", "me", "we", "us", "you", "he", "him", "she", "they", "them",
    "it", "who", "whom", "what", "which", "whose", "where", "when",
    "how", "why", "myself", "yourself", "himself", "herself", "itself",
    "ourselves", "themselves",
    # ---- prepositions / conjunctions ----
    "in", "on", "at", "to", "for", "of", "with", "by", "from", "up",
    "about", "into", "through", "during", "before", "after", "above",
    "below", "between", "under", "over", "out", "off", "down", "along",
    "around", "among", "against", "without", "within", "upon", "across",
    "behind", "beyond", "near", "toward", "towards",
    "and", "but", "or", "so", "yet", "because", "since", "while",
    "although", "though", "if", "unless", "until", "whether", "than",
    "as", "once", "either", "neither",
    # ---- common verbs / auxiliaries ----
    "is", "am", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having", "do", "does", "did", "doing",
    "will", "would", "shall", "should", "may", "might", "must", "can",
    "could", "need", "dare", "ought",
    "get", "got", "gets", "getting",
    "let", "make", "made", "say", "said",
    # ---- common adverbs / adjectives ----
    "very", "too", "also", "just", "then", "now", "here", "there",
    "when", "where", "why", "how", "well", "back", "even", "still",
    "already", "always", "never", "often", "sometimes", "usually",
    "again", "further", "thus", "hence", "therefore", "however",
    "moreover", "otherwise", "instead", "rather", "quite", "almost",
    "enough", "really", "perhaps", "certainly", "simply", "actually",
    # ---- other high-frequency English words ----
    "one", "two", "first", "new", "like", "time", "way", "may",
    "people", "know", "take", "come", "think", "see", "look",
    "want", "give", "use", "tell", "work", "call", "try",
    "ask", "seem", "feel", "leave", "put", "mean", "keep",
    "help", "show", "turn", "play", "run", "move", "live",
    "point", "part", "number", "right", "set",
    # ---- academic / textbook filler ----
    "figure", "fig", "table", "page", "chapter", "section", "example",
    "equation", "see", "also", "given", "find", "shown", "following",
    "using", "used", "note", "result", "results", "case", "cases",
    "consider", "determine", "solution", "problem", "problems",
    "assume", "assumed", "respectively", "therefore", "obtained",
    "defined", "definition", "expressed", "expression", "written",
    "referred", "reference", "references", "discussed", "discussion",
    "described", "denoted", "called", "known", "various", "several",
    "respectively", "typically", "approximately", "corresponding",
}

# Regex used to split text into word-like tokens
_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    """Split *text* on whitespace / punctuation and lowercase every token."""
    return _TOKEN_RE.findall(text.lower())


def _extract_keywords(text: str, top_n: int) -> list[str]:
    """Return the *top_n* TF-based keywords from *text*.

    Pipeline:
      1. Tokenize (split on whitespace & punctuation, lowercase).
      2. Remove stopwords.
      3. Remove tokens shorter than 3 characters.
      4. Remove purely-numeric tokens.
      5. Count term frequencies and return the top *top_n*.
    """
    tokens = _tokenize(text)

    filtered = [
        tok for tok in tokens
        if tok not in STOPWORDS
        and len(tok) >= 3
        and not tok.isdigit()
    ]

    counts = Counter(filtered)
    return [word for word, _ in counts.most_common(top_n)]


def _load_sections(sections_dir: Path) -> list[dict[str, Any]]:
    """Load every ``*.json`` file from *sections_dir* and return a list of
    section dicts sorted by ``start_page`` (ascending)."""
    sections: list[dict[str, Any]] = []

    if not sections_dir.is_dir():
        print(f"[INDEX] Warning: sections directory not found — {sections_dir}")
        return sections

    for json_path in sorted(sections_dir.glob("*.json")):
        with open(json_path, "r", encoding="utf-8") as fh:
            section = json.load(fh)
            sections.append(section)

    # Sort by start_page (numerical order)
    sections.sort(key=lambda s: s.get("start_page", 0))
    return sections


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_index(book_name: str, output_dir: str) -> dict[str, Any]:
    """Build and persist a flattened TOC index for the given book.

    Parameters
    ----------
    book_name : str
        Human-readable book name (e.g. ``"Fluid Mechanics Cengel"``).
    output_dir : str
        Base output directory for this book
        (e.g. ``"books/processed/fluid_mechanics/"``).

    Returns
    -------
    dict
        The complete index dictionary (also saved to
        ``<output_dir>/index.json``).
    """
    output_path = Path(output_dir)
    sections_dir = output_path / "sections"

    # 1. Read all section JSON files
    raw_sections = _load_sections(sections_dir)

    # 2. Build per-section entries with TF-based keywords
    index_sections: list[dict[str, Any]] = []
    for sec in raw_sections:
        text: str = sec.get("text", "")
        start_page: int = sec.get("start_page", 0)
        end_page: int = sec.get("end_page", start_page)

        # 1-based page range string for human readability
        page_range = f"{start_page + 1}-{end_page + 1}" if start_page != end_page else str(start_page + 1)

        keywords = _extract_keywords(text, TOP_N_KEYWORDS)

        index_sections.append({
            "section_id": sec.get("section_id", ""),
            "title": sec.get("title", ""),
            "level": sec.get("level", 1),
            "chapter": sec.get("chapter", ""),
            "page_range": page_range,
            "keywords": keywords,
        })

    # 3. Assemble the index
    index: dict[str, Any] = {
        "book_name": book_name,
        "total_sections": len(index_sections),
        "sections": index_sections,
    }

    # 4. Write to disk
    index_path = output_path / "index.json"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with open(index_path, "w", encoding="utf-8") as fh:
        json.dump(index, fh, indent=2, ensure_ascii=False)

    # 5. Log
    print(
        f"[INDEX] Built index for '{book_name}': "
        f"{len(index_sections)} sections, saved to index.json"
    )

    return index


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python -m preprocessing.build_index <book_name> <output_dir>")
        sys.exit(1)

    _book_name = sys.argv[1]
    _output_dir = sys.argv[2]
    build_index(_book_name, _output_dir)
