"""Phase 1: LLM-based section router.

Reads index.json and selects the best-matching section(s) for a user query.

Architecture Overview
---------------------
The router works in three stages:

1. FLATTEN: On first call (or once per session), load the book's index.json
   and flatten its hierarchical chapter→section tree into a single list of
   "routable candidates." Non-content sections (cover, bibliography, back-
   matter index pages A-Z) are excluded here — cheaply, before any LLM call.

2. RETRIEVE: Use local TF-IDF cosine similarity to shortlist a small set of
   likely sections. This runs offline and costs no tokens.

3. ROUTE: Build a compact textual "catalog" from only the shortlist, then ask
   an LLM to pick the top-K best-matching section(s). The LLM returns
   structured JSON: section_id + confidence + reason.

4. ENRICH (optional): Cross-reference the top-1 result against the book's
   confusable_pairs.json. If the top-1 section has a known confusable sibling,
   log a warning (for eval diagnostics). In a future iteration this could auto-
   inject the sibling into the result list — but for now, logging only.

Why retrieve before routing?
   Larger books can have hundreds of sections, which makes a full-catalog
   prompt too large for low-TPM providers. Local retrieval keeps the LLM prompt
   bounded by the shortlist size instead of the book size.

Usage
-----
    from core.router import route_query, load_book_index

    index = load_book_index("ai")
    results = route_query("What is the Bellman equation?", index, top_k=3)
    for r in results:
        print(r["section_id"], r["confidence"], r["reason"])
"""

from __future__ import annotations

import json
import logging
import re

from config.settings import (
    BOOKS_PROCESSED_PATH,
    GROQ_LLM_API_KEY,
    GROQ_LLM_MODEL_NAME,
    ROUTER_LLM_TIMEOUT_SECONDS,
    ROUTER_SHORTLIST_N,
    ROUTER_TOP_K,
)
from core.retrieval import shortlist_candidates

# ---------------------------------------------------------------------------
# Logging — all router activity goes through this logger so you can control
# verbosity via standard Python logging config (e.g. DEBUG for prompt/response
# inspection during eval, WARNING for production).
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LLM provider base URLs — same pattern as preprocessing/build_index.py.
# We reuse the Groq OpenAI-compatible endpoint; Gemini can be added back
# by uncommenting the line below and adding it to the providers list.
# ---------------------------------------------------------------------------
_BASE_URLS: dict[str, str] = {
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "groq": "https://api.groq.com/openai/v1",
}

# ---------------------------------------------------------------------------
# Router-specific defaults — importable from config/settings.py if overridden,
# but sane defaults are set here so the router works out of the box.
# ---------------------------------------------------------------------------
# How many top candidates the router returns by default.
# Overridable by the top_k parameter in route_query().
# Which LLM model the router uses. Defaults to the same Groq model used by
# build_index.py — no new provider config needed.
try:
    from config.settings import ROUTER_LLM_MODEL_NAME
except ImportError:
    ROUTER_LLM_MODEL_NAME = GROQ_LLM_MODEL_NAME


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 1: INDEX LOADING & FLATTENING
# ═══════════════════════════════════════════════════════════════════════════

# --- Denylist: section IDs that are structural/non-content and should never
#     be router candidates. These are front matter, back matter, and the
#     alphabetical index pages (a-z) that some books include.
_DENYLIST_IDS: set[str] = {
    "cover", "title-page", "copyright", "contents", "bibliography",
    "front-matter", "acknowledgments",
}

# --- Title patterns for "deprioritized" sections. These are kept in the
#     candidate list (a student might ask "give me practice exercises on X"),
#     but are marked so the prompt can tell the LLM to prefer topical content.
_DEPRIORITIZED_TITLE_RE: re.Pattern = re.compile(
    r"summary|bibliographical\s+and\s+historical\s+notes|exercises|problems",
    re.IGNORECASE,
)


def _is_single_letter_id(section_id: str) -> bool:
    """Check if section_id is a single letter a-z (back-matter index page).

    These are the alphabetical index pages some textbooks include (A.json,
    B.json, ...) — they're structurally present in index.json but contain
    only name/page-number pairs, never topical content a student would
    ask about.
    """
    return len(section_id) == 1 and section_id.isalpha()


def _should_exclude(section_id: str) -> bool:
    """Return True if this section should be entirely excluded from routing.

    Excluded sections:
    - Single-letter IDs (a-z): back-matter index pages
    - IDs in the explicit denylist: cover, title-page, copyright, etc.

    Note: summary/exercises sections are NOT excluded, just deprioritized
    (see _is_deprioritized). A student might legitimately ask for practice
    questions on a topic, and that maps to an exercises section.
    """
    return _is_single_letter_id(section_id) or section_id in _DENYLIST_IDS


def _is_deprioritized(title: str) -> bool:
    """Return True if this section's title matches a deprioritized pattern.

    Deprioritized sections stay in the candidate list but are explicitly
    flagged in the LLM prompt as lower-priority, so the LLM prefers
    topical content sections over summary/exercises unless the query
    specifically asks for them.
    """
    return bool(_DEPRIORITIZED_TITLE_RE.search(title))


def load_book_index(book_name: str) -> dict:
    """Load a book's index.json from the processed books directory.

    Args:
        book_name: Folder name under books/processed/ (e.g. "ai", "coa").

    Returns:
        The parsed index.json dict with keys: book_name, total_sections, chapters.

    Raises:
        FileNotFoundError: If the index.json doesn't exist for this book.
    """
    index_path = BOOKS_PROCESSED_PATH / book_name / "index.json"
    if not index_path.exists():
        raise FileNotFoundError(
            f"No index.json found for book '{book_name}' at {index_path}. "
            f"Has Phase 0 preprocessing been run for this book?"
        )
    with open(index_path, "r", encoding="utf-8") as f:
        return json.load(f)


def flatten_index(index: dict) -> list[dict]:
    """Walk the chapters tree and produce a flat list of routable candidates.

    This is the core pre-processing step before any LLM call. It:
    1. Recursively traverses the hierarchical chapters→children tree
    2. Propagates parent/chapter context down to every leaf (so the LLM sees
       "Chapter 17: Making Complex Decisions → 17.2 Value Iteration" instead
       of just "17.2 Value Iteration")
    3. Excludes non-content sections (cover, bibliography, index pages a-z)
    4. Flags summary/exercises sections as deprioritized

    Args:
        index: The parsed index.json dict.

    Returns:
        Flat list of dicts, each with keys:
        - section_id: str — unique identifier matching the section file name
        - title: str — section heading
        - level: int — heading level (1=chapter, 2=section, 3=subsection)
        - page_range: str — e.g. "684-696"
        - tfidf_keywords: list[str] — discriminative keywords for this section
        - keywords: list[str] — raw frequency keywords (kept for reference,
                                 not used in the router prompt to avoid noise)
        - abstract: str — LLM-generated or text-based summary
        - has_visuals: bool — whether the section contains diagrams/images
        - chapter_title: str — title of the top-level chapter this belongs to
        - parent_title: str — title of the immediate parent (= chapter_title
                               for direct children, or intermediate heading
                               for deeper nesting)
        - is_deprioritized: bool — True for summary/exercises/notes sections
    """
    flat: list[dict] = []

    def _walk(
        nodes: list[dict],
        chapter_title: str = "",
        parent_title: str = "",
    ) -> None:
        """Recursively walk the tree, carrying chapter/parent context down.

        Args:
            nodes: List of section dicts at this level.
            chapter_title: Title of the top-level chapter ancestor.
            parent_title: Title of the immediate parent node.
        """
        for node in nodes:
            sid = node.get("section_id", "")
            title = node.get("title", "")

            # --- Determine chapter title for this node ---
            # If this is a top-level node (no chapter_title passed down yet),
            # it IS the chapter — use its own title as the chapter context.
            current_chapter = chapter_title if chapter_title else title

            # --- Exclusion check ---
            if _should_exclude(sid):
                logger.debug("Excluding section '%s' (%s)", sid, title)
                # Still recurse into children — some excluded nodes (like
                # "contents") might have content-bearing children in other
                # books. Unlikely for current data, but defensive.
                _walk(
                    node.get("children", []),
                    chapter_title=current_chapter,
                    parent_title=title,
                )
                continue

            # --- Build the flattened entry ---
            entry = {
                "section_id": sid,
                "title": title,
                "level": node.get("level", 0),
                "page_range": node.get("page_range", ""),
                "tfidf_keywords": node.get("tfidf_keywords", []),
                "keywords": node.get("keywords", []),
                "abstract": node.get("abstract", ""),
                "has_visuals": node.get("has_visuals", False),
                "chapter_title": current_chapter,
                "parent_title": parent_title if parent_title else current_chapter,
                "is_deprioritized": _is_deprioritized(title),
            }
            flat.append(entry)

            # --- Recurse into children ---
            _walk(
                node.get("children", []),
                chapter_title=current_chapter,
                parent_title=title,
            )

    _walk(index.get("chapters", []))

    logger.info(
        "Flattened index: %d candidates from %d total sections "
        "(excluded %d non-content sections)",
        len(flat),
        index.get("total_sections", 0),
        index.get("total_sections", 0) - len(flat),
    )
    return flat


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 2: CONFUSABLE PAIRS
# ═══════════════════════════════════════════════════════════════════════════

def load_confusable_map(book_name: str) -> dict[str, list[str]]:
    """Load confusable_pairs.json and build a section_id → confusable siblings map.

    Phase 0 generates confusable_pairs.json identifying sections with high
    keyword overlap (e.g. value-iteration ↔ policy-iteration). This function
    loads that data and indexes it as a lookup map so the router can check
    whether its top-1 pick has known confusable siblings.

    Current usage: logging only during evaluation. If eval shows the router
    actually fails on flagged pairs, this can be promoted to automatic
    candidate injection (bumping the confused sibling into the result list).

    Args:
        book_name: Folder name under books/processed/ (e.g. "ai").

    Returns:
        Dict mapping section_id → list of section_ids it's commonly confused
        with. Returns empty dict if the file doesn't exist or can't be parsed.

    Example:
        >>> cmap = load_confusable_map("ai")
        >>> cmap.get("17-2-value-iteration")
        ["17-3-policy-iteration"]
    """
    pairs_path = BOOKS_PROCESSED_PATH / book_name / "confusable_pairs.json"

    if not pairs_path.exists():
        logger.warning(
            "No confusable_pairs.json found for book '%s' — "
            "confusable-pair checking disabled.",
            book_name,
        )
        return {}

    try:
        with open(pairs_path, "r", encoding="utf-8") as f:
            pairs = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning(
            "Failed to load confusable_pairs.json for '%s': %s",
            book_name,
            exc,
        )
        return {}

    # Build the bidirectional map: if (A, B) is a confusable pair,
    # then A maps to [B] and B maps to [A].
    confusable_map: dict[str, list[str]] = {}

    # Deduplicate pairs — the file contains duplicates (each pair appears
    # twice in the AI book's data).
    seen_pairs: set[tuple[str, str]] = set()

    for pair in pairs:
        a = pair.get("section_a", "")
        b = pair.get("section_b", "")
        if not a or not b:
            continue

        # Skip single-letter pairs (index pages) — they're already excluded
        # from routing, no point tracking their confusability.
        if _is_single_letter_id(a) or _is_single_letter_id(b):
            continue

        # Deduplicate: normalize pair order
        canonical = (min(a, b), max(a, b))
        if canonical in seen_pairs:
            continue
        seen_pairs.add(canonical)

        # Add bidirectional entries
        confusable_map.setdefault(a, []).append(b)
        confusable_map.setdefault(b, []).append(a)

    logger.info(
        "Loaded confusable map for '%s': %d sections have known confusable siblings.",
        book_name,
        len(confusable_map),
    )
    return confusable_map


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 3: LLM INTERACTION
# ═══════════════════════════════════════════════════════════════════════════

def _call_router_llm(prompt: str) -> str | None:
    """Call the LLM for routing. Reuses the same OpenAI-compatible pattern
    as preprocessing/build_index.py's _call_llm().

    Uses the Groq endpoint by default. Returns the raw response text,
    or None if the call fails.

    Why a separate function from build_index._call_llm?
    Because the router lives in core/, not preprocessing/. We don't want
    a runtime import dependency from core → preprocessing (that direction
    should never exist). The pattern is identical though — if you change
    one, consider changing the other.
    """
    try:
        from openai import OpenAI  # noqa: E402 — deferred import
    except ImportError:
        logger.error(
            "The 'openai' package is required for LLM routing. "
            "Install it: pip install 'openai>=1.0.0'"
        )
        return None

    # Validate that we have credentials
    if not GROQ_LLM_API_KEY or not ROUTER_LLM_MODEL_NAME:
        logger.error(
            "GROQ_LLM_API_KEY and ROUTER_LLM_MODEL_NAME must be set in .env "
            "for LLM routing to work."
        )
        return None

    providers = [
        ("groq", GROQ_LLM_API_KEY, ROUTER_LLM_MODEL_NAME),
    ]

    for name, api_key, model_name in providers:
        try:
            client = OpenAI(
                api_key=api_key,
                base_url=_BASE_URLS[name],
                timeout=ROUTER_LLM_TIMEOUT_SECONDS,
                max_retries=0,
            )
            resp = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                # Low temperature for routing — we want deterministic, precise
                # section selection, not creative prose.
                temperature=0.2,
                max_tokens=1000,
            )
            return resp.choices[0].message.content
        except Exception as exc:
            logger.warning(
                "Router LLM call failed (%s / %s): %s",
                name,
                model_name,
                exc,
            )
            continue

    return None


def _parse_llm_response(raw: str) -> list[dict]:
    """Parse the LLM's JSON response into a list of section picks.

    Handles two common LLM quirks:
    1. Response wrapped in markdown code fences (```json ... ```)
    2. Response is a bare JSON object instead of the expected structure

    Expected format from the LLM:
        {"sections": [
            {"section_id": "...", "confidence": "high|medium|low", "reason": "..."},
            ...
        ]}

    Returns:
        List of dicts with keys: section_id, confidence, reason.
        Returns empty list if parsing fails entirely.
    """
    if not raw:
        return []

    # Step 1: Strip markdown code fences if present
    # (Same pattern as build_index.py's abstract parser)
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned)

    # Step 2: Try to parse as JSON
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        # Sometimes the LLM returns slightly malformed JSON — try to find
        # the first { and last } and parse just that substring.
        try:
            start = cleaned.index("{")
            end = cleaned.rindex("}") + 1
            parsed = json.loads(cleaned[start:end])
        except (ValueError, json.JSONDecodeError) as exc:
            logger.error(
                "Failed to parse router LLM response as JSON: %s\n"
                "Raw response (first 500 chars): %s",
                exc,
                raw[:500],
            )
            return []

    # Step 3: Extract the sections list
    if isinstance(parsed, dict) and "sections" in parsed:
        sections = parsed["sections"]
    elif isinstance(parsed, list):
        # LLM returned a bare list instead of {"sections": [...]}
        sections = parsed
    else:
        logger.error(
            "Router LLM response has unexpected structure. "
            "Expected {'sections': [...]}, got keys: %s",
            list(parsed.keys()) if isinstance(parsed, dict) else type(parsed),
        )
        return []

    # Step 4: Validate each section entry
    validated: list[dict] = []
    for s in sections:
        if not isinstance(s, dict) or "section_id" not in s:
            logger.warning("Skipping malformed section entry: %s", s)
            continue
        validated.append({
            "section_id": s["section_id"],
            "confidence": s.get("confidence", "unknown"),
            "reason": s.get("reason", ""),
        })

    return validated


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 4: PROMPT CONSTRUCTION
# ═══════════════════════════════════════════════════════════════════════════

def _build_catalog(candidates: list[dict]) -> str:
    """Build the textual catalog of section candidates for the LLM prompt.

    Each candidate is represented as a compact one-entry block:
        [section_id] Chapter: <chapter_title> | <title>
        Abstract: <first ~150 chars of abstract>
        Keywords: <tfidf_keywords, comma-separated>

    Why this format?
    - Compact enough for the small Stage A shortlist sent to the LLM
    - Gives the LLM three orthogonal signals: structural context (chapter),
      semantic content (abstract), and lexical features (TF-IDF keywords)
    - Omits raw 'keywords' field (frequency-based, noisier than TF-IDF)
      to save tokens and reduce confusion

    Why truncate abstracts?
    - Full abstracts can be 200-400 chars each. Truncating to ~200 chars keeps
      the shortlist prompt comfortably under low-tier TPM limits while
      preserving the most informative opening of each abstract.

    Args:
        candidates: Flattened list from flatten_index().

    Returns:
        Multi-line string, one block per candidate.
    """
    lines: list[str] = []
    for c in candidates:
        # --- Section header line ---
        # Format: [section_id] Chapter: X | Section Title
        # The [brackets] around section_id make it easy for the LLM to
        # extract and return the exact ID in its response.
        header = f"[{c['section_id']}] Chapter: {c['chapter_title']} | {c['title']}"

        # --- Abstract snippet ---
        # Take first ~200 chars of the abstract. If the abstract has the
        # "Description | Key concepts: ... | Example questions: ..." format,
        # we want at least the description part.
        abstract = c.get("abstract", "")
        # Truncate at a word boundary near 200 chars
        if len(abstract) > 200:
            truncated = abstract[:200]
            # Don't cut mid-word
            last_space = truncated.rfind(" ")
            if last_space > 150:
                truncated = truncated[:last_space]
            abstract = truncated + "..."

        # --- TF-IDF keywords ---
        tfidf = ", ".join(c.get("tfidf_keywords", [])[:10])

        # --- Deprioritization marker ---
        # If this is a summary/exercises section, add a marker so the prompt
        # can instruct the LLM accordingly.
        depri = " [SUPPLEMENTARY]" if c.get("is_deprioritized", False) else ""

        lines.append(f"{header}{depri}")
        lines.append(f"  Abstract: {abstract}")
        if tfidf:
            lines.append(f"  Keywords: {tfidf}")
        lines.append("")  # blank line between entries

    return "\n".join(lines)


def _build_router_prompt(question: str, catalog: str, top_k: int) -> str:
    """Construct the full routing prompt sent to the LLM.

    Prompt design principles (from the Phase 1 plan):
    1. Feed abstract + chapter_title + title as primary signal
    2. Ask for structured JSON output (not prose)
    3. Explicitly allow multi-section answers for comparison questions
    4. Instruct to prefer precision over recall — return both when uncertain
    5. Include confidence field for downstream Phase 2 consumption

    The prompt includes 2 examples showing:
    - Single-section answer (factual lookup)
    - Multi-section answer (comparison question)

    Args:
        question: The student's question.
        catalog: Output of _build_catalog().
        top_k: Maximum number of sections to return.

    Returns:
        The complete prompt string.
    """
    return f"""You are a section router for an engineering textbook study assistant.

TASK: Given a student's question, identify the {top_k} most relevant textbook section(s) from the catalog below. Return structured JSON only.

RULES:
1. Return 1 to {top_k} sections, ordered by relevance (best match first).
2. For comparison or cross-topic questions (e.g. "compare X and Y"), return multiple sections — one for each topic being compared.
3. If uncertain between two adjacent/overlapping sections, return BOTH with confidence "medium" rather than guessing one. It is better to return an extra relevant section than to miss the right one.
4. Sections marked [SUPPLEMENTARY] contain summaries, exercises, or historical notes. Only select these if the student specifically asks for practice problems, exercises, or historical context. Prefer the main topical section otherwise.
5. Match based on conceptual relevance, not just keyword overlap. A question about "Bellman optimality" maps to the section discussing that concept, even if the exact phrase doesn't appear in the section title.

OUTPUT FORMAT (JSON only, no prose before or after):
{{
  "sections": [
    {{
      "section_id": "<exact section_id from the catalog>",
      "confidence": "high|medium|low",
      "reason": "<1 sentence explaining why this section matches>"
    }}
  ]
}}

EXAMPLES:

Student question: "What is the Bellman equation used for in MDPs?"
Good answer: {{"sections": [{{"section_id": "17-2-value-iteration", "confidence": "high", "reason": "This section covers value iteration which centers on the Bellman equation for MDPs."}}]}}

Student question: "Compare value iteration and policy iteration"
Good answer: {{"sections": [{{"section_id": "17-2-value-iteration", "confidence": "high", "reason": "Covers value iteration algorithm and its convergence."}}, {{"section_id": "17-3-policy-iteration", "confidence": "high", "reason": "Covers policy iteration as an alternative to value iteration."}}]}}

--- CATALOG START ---
{catalog}
--- CATALOG END ---

STUDENT QUESTION: {question}
"""


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 5: MAIN ROUTING FUNCTION
# ═══════════════════════════════════════════════════════════════════════════

def route_query(
    question: str,
    index: dict,
    top_k: int = ROUTER_TOP_K,
    book_name: str | None = None,
    confusable_map: dict[str, list[str]] | None = None,
) -> list[dict]:
    """Route a student's question to the best-matching textbook section(s).

    This is the main entry point for Phase 1. Given a student question and
    a book's parsed index.json, it:
    1. Flattens the index tree into routable candidates
    2. Shortlists candidates locally with TF-IDF retrieval
    3. Builds a compact catalog prompt from the shortlist
    4. Calls the LLM to select the top-K sections
    5. Enriches results with metadata (page_range, has_visuals, chapter_title)
    6. Logs confusable-pair warnings if applicable

    Args:
        question: The student's natural-language question.
        index: Parsed index.json (from load_book_index()).
        top_k: Maximum number of sections to return (default: 3).
        book_name: Optional book name for confusable-pair logging.
                   If None, confusable checking is skipped.
        confusable_map: Pre-loaded confusable map (from load_confusable_map()).
                        If None and book_name is provided, it will be loaded
                        on the fly. Pass it in to avoid re-reading the file
                        on every call.

    Returns:
        List of result dicts, ordered by LLM ranking (best match first).
        Each dict contains:
        - section_id: str
        - title: str
        - chapter_title: str
        - page_range: str
        - has_visuals: bool
        - confidence: str ("high", "medium", "low", or "unknown")
        - reason: str (LLM's explanation for the match)
        - is_deprioritized: bool

    Returns empty list if the LLM call fails or returns no valid sections.
    """
    # ── Step 1: Flatten the index ──
    candidates = flatten_index(index)
    if not candidates:
        logger.error("No routable candidates after flattening index.")
        return []

    # Build a lookup map for enriching LLM results with full metadata
    # Key = section_id, Value = the full candidate dict
    candidates_map: dict[str, dict] = {c["section_id"]: c for c in candidates}

    # ── Step 1.5: Local retrieval shortlist ──
    # This offline Stage A keeps the Stage B LLM prompt bounded by
    # ROUTER_SHORTLIST_N rather than total book size.
    shortlisted = shortlist_candidates(
        question=question,
        candidates=candidates,
        top_n=ROUTER_SHORTLIST_N,
    )

    # ── Step 2: Build catalog and prompt from the shortlist ──
    catalog = _build_catalog(shortlisted)
    if len(catalog) > 20000:
        logger.warning(
            "Router catalog is %d chars (~%d tokens) even after shortlisting; "
            "consider lowering ROUTER_SHORTLIST_N.",
            len(catalog),
            len(catalog) // 4,
        )

    prompt = _build_router_prompt(question, catalog, top_k)

    logger.debug(
        "Router prompt built: %d/%d candidates, %d chars total prompt",
        len(shortlisted),
        len(candidates),
        len(prompt),
    )

    # ── Step 3: Call the LLM ──
    raw_response = _call_router_llm(prompt)
    if not raw_response:
        logger.error("Router LLM returned no response.")
        return []

    logger.debug("Router LLM raw response:\n%s", raw_response[:500])

    # ── Step 4: Parse the response ──
    llm_picks = _parse_llm_response(raw_response)
    if not llm_picks:
        logger.error("Router LLM response contained no valid section picks.")
        return []

    # ── Step 5: Enrich results with full metadata from the index ──
    results: list[dict] = []
    for pick in llm_picks[:top_k]:
        sid = pick["section_id"]

        if sid not in candidates_map:
            # The LLM returned a section_id that doesn't exist in our
            # candidate list. This can happen if the LLM hallucinates an ID
            # or returns an excluded section. Log and skip.
            logger.warning(
                "Router returned unknown section_id '%s' — skipping. "
                "This may indicate the LLM hallucinated an ID.",
                sid,
            )
            continue

        candidate = candidates_map[sid]
        results.append({
            "section_id": sid,
            "title": candidate["title"],
            "chapter_title": candidate["chapter_title"],
            "page_range": candidate["page_range"],
            "has_visuals": candidate["has_visuals"],
            "confidence": pick["confidence"],
            "reason": pick["reason"],
            "is_deprioritized": candidate.get("is_deprioritized", False),
        })

    # ── Step 6: Confusable-pair logging ──
    # If the top-1 result has a known confusable sibling, log a warning.
    # This is pure diagnostics for eval — it doesn't modify the results.
    if results and book_name:
        # Load confusable map if not provided
        if confusable_map is None:
            confusable_map = load_confusable_map(book_name)

        top_sid = results[0]["section_id"]
        if top_sid in confusable_map:
            siblings = confusable_map[top_sid]
            # Check if any sibling is already in the results
            result_sids = {r["section_id"] for r in results}
            missing_siblings = [s for s in siblings if s not in result_sids]

            if missing_siblings:
                logger.warning(
                    "CONFUSABLE PAIR ALERT: Top-1 result '%s' has confusable "
                    "siblings %s that are NOT in the result list. "
                    "Question: '%s'. This may be a routing error — check "
                    "the eval misses file.",
                    top_sid,
                    missing_siblings,
                    question[:100],
                )
            else:
                logger.debug(
                    "Confusable siblings for '%s' already in results: %s",
                    top_sid,
                    [s for s in siblings if s in result_sids],
                )

    logger.info(
        "Route result for '%s': %s",
        question[:80],
        [(r["section_id"], r["confidence"]) for r in results],
    )
    return results


# ═══════════════════════════════════════════════════════════════════════════
# SECTION 6: CONVENIENCE / HIGH-LEVEL API
# ═══════════════════════════════════════════════════════════════════════════

def route(book_name: str, question: str, top_k: int = ROUTER_TOP_K) -> list[dict]:
    """Convenience wrapper: load index + route in one call.

    This is the simplest way to use the router. For repeated queries against
    the same book, prefer loading the index once with load_book_index() and
    calling route_query() directly to avoid re-reading the file.

    Args:
        book_name: Book folder name (e.g. "ai", "coa").
        question: Student's question.
        top_k: Max sections to return.

    Returns:
        Same as route_query().
    """
    index = load_book_index(book_name)
    confusable_map = load_confusable_map(book_name)
    return route_query(
        question=question,
        index=index,
        top_k=top_k,
        book_name=book_name,
        confusable_map=confusable_map,
    )
