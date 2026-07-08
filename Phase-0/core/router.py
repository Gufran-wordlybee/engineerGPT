"""
core.router — LLM-based section router with two-stage routing and fallback.

Routes a student question to the best-matching section(s) in a preprocessed
textbook.  This is the core of Phase 1: given a question and a book name,
it returns which section(s) the answer lives in.

Architecture
------------
For large books (> ``TWO_STAGE_THRESHOLD`` sections):

    question ──► Stage 1: chapter pick (1–2 chapters)
                     │   (only titles + abstracts, cheap prompt)
                     ▼
                 Stage 2: section pick within those chapters
                     │   (full subtree + confusable-pair disambiguation)
                     ▼
                 Parse JSON → validate section_ids → RouterResult

For small books (≤ threshold):

    question ──► Single flat prompt (all sections + confusable pairs)
                     │
                     ▼
                 Parse JSON → validate section_ids → RouterResult

Fallback chain:
    1. LLM two-stage / single-stage
    2. LLM retry with stricter prompt (on parse failure)
    3. TF-IDF keyword overlap (no LLM needed)

Public API
----------
route(book_name, question, top_k) -> RouterResult
"""

from __future__ import annotations

import json
import re
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, TypedDict

from config.settings import (
    BOOKS_PROCESSED_PATH,
    ROUTER_MODEL_NAME,
    ROUTER_TEMPERATURE,
    ROUTER_TOP_K_DEFAULT,
    TWO_STAGE_THRESHOLD,
    LLM_API_KEY,
)
from core.llm_client import call_llm


# ═══════════════════════════════════════════════════════════════════════════
# Types
# ═══════════════════════════════════════════════════════════════════════════

class RouterResult(TypedDict):
    """Result of routing a question to book sections.

    Attributes
    ----------
    section_ids : list[str]
        Ordered list of section IDs, best match first.
        Example: ``["3.2", "3.2.1"]``
    confidence : str
        ``"high"`` if the LLM is confident in the match,
        ``"low"`` if uncertain or using fallback.
    reasoning : str
        Short explanation for debugging and eval logs.
    """
    section_ids: list[str]
    confidence: str   # "high" | "low"
    reasoning: str


# ═══════════════════════════════════════════════════════════════════════════
# Index loading & caching
# ═══════════════════════════════════════════════════════════════════════════

# Simple module-level cache so repeated calls don't re-read from disk.
_index_cache: dict[str, dict[str, Any]] = {}
_confusable_cache: dict[str, list[dict[str, Any]]] = {}


def _load_index(book_name: str) -> dict[str, Any]:
    """Load a book's ``index.json`` from the processed books directory.

    Caches the result in memory so subsequent calls for the same book
    don't hit the filesystem.

    Parameters
    ----------
    book_name : str
        The snake_case book directory name under ``books/processed/``.

    Returns
    -------
    dict
        The full index with ``chapters``, ``total_sections``, etc.

    Raises
    ------
    FileNotFoundError
        If the index file doesn't exist (book not preprocessed).
    """
    if book_name in _index_cache:
        return _index_cache[book_name]

    index_path = Path(BOOKS_PROCESSED_PATH) / book_name / "index.json"
    if not index_path.exists():
        raise FileNotFoundError(
            f"No index.json found for book '{book_name}' at {index_path}. "
            f"Run: python -m preprocessing.run_pipeline books/raw/<book>.pdf"
        )

    with open(index_path, "r", encoding="utf-8") as fh:
        index = json.load(fh)

    _index_cache[book_name] = index
    return index


def _load_confusable_pairs(book_name: str) -> list[dict[str, Any]]:
    """Load ``confusable_pairs.json`` for a book, if it exists.

    Returns an empty list if the file doesn't exist (no confusable pairs
    were detected during preprocessing).
    """
    if book_name in _confusable_cache:
        return _confusable_cache[book_name]

    conf_path = Path(BOOKS_PROCESSED_PATH) / book_name / "confusable_pairs.json"
    if not conf_path.exists():
        _confusable_cache[book_name] = []
        return []

    with open(conf_path, "r", encoding="utf-8") as fh:
        pairs = json.load(fh)

    _confusable_cache[book_name] = pairs
    return pairs


# ═══════════════════════════════════════════════════════════════════════════
# Index traversal helpers
# ═══════════════════════════════════════════════════════════════════════════

def _count_all_sections(index: dict[str, Any]) -> int:
    """Count every section node in the tree (including nested children).

    Used to decide between single-stage and two-stage routing.
    """
    count = 0

    def _walk(nodes: list[dict[str, Any]]) -> None:
        nonlocal count
        for node in nodes:
            count += 1
            if node.get("children"):
                _walk(node["children"])

    _walk(index.get("chapters", []))
    return count


def _collect_all_section_ids(index: dict[str, Any]) -> set[str]:
    """Collect every ``section_id`` in the index tree into a flat set.

    Used to validate that LLM-returned section_ids actually exist.
    """
    ids: set[str] = set()

    def _walk(nodes: list[dict[str, Any]]) -> None:
        for node in nodes:
            sid = node.get("section_id", "")
            if sid:
                ids.add(sid)
            if node.get("children"):
                _walk(node["children"])

    _walk(index.get("chapters", []))
    return ids


def _collect_all_sections_flat(index: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten the tree into a list of all section nodes (for keyword fallback).

    Each returned dict has at least: section_id, title, tfidf_keywords.
    """
    result: list[dict[str, Any]] = []

    def _walk(nodes: list[dict[str, Any]]) -> None:
        for node in nodes:
            result.append(node)
            if node.get("children"):
                _walk(node["children"])

    _walk(index.get("chapters", []))
    return result


def _get_chapter_subtrees(
    index: dict[str, Any], chapter_ids: list[str]
) -> list[dict[str, Any]]:
    """Return the full subtrees for the given chapter section_ids.

    Used in Stage 2 to send only the relevant chapters' children to
    the LLM, keeping the prompt small.
    """
    subtrees: list[dict[str, Any]] = []
    for chapter in index.get("chapters", []):
        if chapter.get("section_id") in chapter_ids:
            subtrees.append(chapter)
    return subtrees


# ═══════════════════════════════════════════════════════════════════════════
# Confusable-pair injection
# ═══════════════════════════════════════════════════════════════════════════

def _build_confusable_context(
    candidate_section_ids: list[str],
    confusable_pairs: list[dict[str, Any]],
    index: dict[str, Any],
) -> str:
    """Build disambiguation text for sections involved in confusable pairs.

    When a candidate section appears in a confusable pair, we inject a
    warning like:

        "Section 3.2 (Fluid Statics) is commonly confused with
         Section 3.3 (Fluid Dynamics). Shared keywords: pressure, force.
         Only pick 3.2 if the question is specifically about STATIC fluids."

    This directly targets the #1 hard requirement: no wrong-topic answers.

    Parameters
    ----------
    candidate_section_ids : list[str]
        Section IDs that are candidates (from Stage 1 chapters, or all).
    confusable_pairs : list[dict]
        From ``confusable_pairs.json``.
    index : dict
        The full book index (used to look up section titles).

    Returns
    -------
    str
        Disambiguation text to inject into the prompt, or empty string.
    """
    if not confusable_pairs:
        return ""

    # Build a quick section_id → title lookup
    title_map: dict[str, str] = {}

    def _walk(nodes: list[dict[str, Any]]) -> None:
        for node in nodes:
            title_map[node.get("section_id", "")] = node.get("title", "")
            if node.get("children"):
                _walk(node["children"])

    _walk(index.get("chapters", []))

    candidate_set = set(candidate_section_ids)
    lines: list[str] = []

    for pair in confusable_pairs:
        a = pair["section_a"]
        b = pair["section_b"]
        shared = pair.get("shared_keywords", [])

        # Only inject if at least one side is a candidate
        if a not in candidate_set and b not in candidate_set:
            continue

        title_a = title_map.get(a, a)
        title_b = title_map.get(b, b)
        shared_str = ", ".join(shared[:8])  # cap at 8 keywords for brevity

        lines.append(
            f"⚠ DISAMBIGUATION: Section '{a}' ({title_a}) is commonly "
            f"confused with section '{b}' ({title_b}). "
            f"Shared keywords: {shared_str}. "
            f"Only pick '{a}' if the question specifically matches its "
            f"unique content, not the shared keywords."
        )

    if not lines:
        return ""

    return (
        "\n\n--- CONFUSABLE SECTION WARNINGS ---\n"
        + "\n".join(lines)
        + "\n--- END WARNINGS ---\n"
    )


# ═══════════════════════════════════════════════════════════════════════════
# Prompt builders
# ═══════════════════════════════════════════════════════════════════════════

_JSON_FORMAT_INSTRUCTION = (
    'Respond with ONLY valid JSON in this exact format, no markdown fences:\n'
    '{{"section_ids": ["<id1>", "<id2>"], "confidence": "high" or "low", '
    '"reasoning": "<1-2 sentence explanation>"}}\n'
    'Rules:\n'
    '- section_ids: ordered list, best match first, at most {top_k} items\n'
    '- confidence: "high" if you are sure, "low" if uncertain\n'
    '- reasoning: brief explanation of why you chose these sections\n'
)

_RETRY_INSTRUCTION = (
    "\n\nWARNING: YOUR PREVIOUS RESPONSE WAS INVALID JSON OR CONTAINED "
    "NON-EXISTENT SECTION IDS. This time, respond with ONLY the JSON "
    "object -- no markdown, no explanation outside the JSON. "
    "Use ONLY section IDs from the list above.\n"
)


def _build_chapter_pick_prompt(
    chapters: list[dict[str, Any]],
    question: str,
    top_k: int = 2,
) -> str:
    """Stage 1 prompt: pick 1–2 candidate chapters from top-level list.

    Sends only chapter titles and abstracts — no sub-sections.
    This keeps the prompt small regardless of total book size.
    """
    chapter_list_lines: list[str] = []
    for ch in chapters:
        sid = ch.get("section_id", "?")
        title = ch.get("title", "Untitled")
        abstract = ch.get("abstract", "")
        chapter_list_lines.append(
            f"  - ID: {sid} | Title: {title}\n    Abstract: {abstract}"
        )

    chapter_list = "\n".join(chapter_list_lines)

    return (
        "You are a textbook section router. A student asked a question "
        "about an engineering textbook. Your job is to identify which "
        "CHAPTER(S) most likely contain the answer.\n\n"
        f"STUDENT QUESTION: {question}\n\n"
        f"AVAILABLE CHAPTERS:\n{chapter_list}\n\n"
        "Pick the 1-2 chapters most likely to contain the answer. "
        "If unsure, pick 2 candidates.\n\n"
        + _JSON_FORMAT_INSTRUCTION.format(top_k=top_k)
    )


def _format_section_tree(nodes: list[dict[str, Any]], indent: int = 0) -> str:
    """Recursively format a section tree for the LLM prompt.

    Includes section_id, title, tfidf_keywords, abstract, and has_visuals
    for each node, with indentation showing the hierarchy.
    """
    lines: list[str] = []
    prefix = "  " * indent

    for node in nodes:
        sid = node.get("section_id", "?")
        title = node.get("title", "Untitled")
        keywords = node.get("tfidf_keywords", [])
        abstract = node.get("abstract", "")
        has_visuals = node.get("has_visuals", False)

        kw_str = ", ".join(keywords[:10]) if keywords else "(none)"
        visual_flag = " [HAS VISUALS]" if has_visuals else ""

        lines.append(
            f"{prefix}- ID: {sid} | Title: {title}{visual_flag}\n"
            f"{prefix}  Keywords: {kw_str}\n"
            f"{prefix}  Abstract: {abstract}"
        )

        # Recurse into children
        if node.get("children"):
            lines.append(_format_section_tree(node["children"], indent + 1))

    return "\n".join(lines)


def _build_section_pick_prompt(
    chapter_subtrees: list[dict[str, Any]],
    question: str,
    confusable_context: str,
    top_k: int = 2,
) -> str:
    """Stage 2 prompt: pick specific section(s) within candidate chapters.

    Receives only the subtree(s) of chapters selected in Stage 1,
    keeping the prompt focused and small.
    """
    section_tree = _format_section_tree(chapter_subtrees)

    return (
        "You are a textbook section router. A student asked a question "
        "and we've narrowed it down to these chapter(s). Your job is to "
        "identify the SPECIFIC SECTION(S) that best answer the question.\n\n"
        f"STUDENT QUESTION: {question}\n\n"
        f"SECTIONS IN CANDIDATE CHAPTERS:\n{section_tree}\n"
        f"{confusable_context}\n"
        "Pick the most specific section(s) that answer the question. "
        "Prefer leaf sections over parent chapters. "
        "If the question spans multiple sections, include all relevant ones.\n\n"
        + _JSON_FORMAT_INSTRUCTION.format(top_k=top_k)
    )


def _build_flat_prompt(
    chapters: list[dict[str, Any]],
    question: str,
    confusable_context: str,
    top_k: int = 2,
) -> str:
    """Single-stage prompt for small books (≤ TWO_STAGE_THRESHOLD sections).

    Sends the entire section tree in one prompt since it fits easily.
    """
    section_tree = _format_section_tree(chapters)

    return (
        "You are a textbook section router. A student asked a question "
        "about an engineering textbook. Your job is to identify which "
        "SECTION(S) most likely contain the answer.\n\n"
        f"STUDENT QUESTION: {question}\n\n"
        f"ALL SECTIONS:\n{section_tree}\n"
        f"{confusable_context}\n"
        "Pick the most specific section(s) that answer the question. "
        "Prefer leaf sections over parent chapters. "
        "If the question spans multiple sections, include all relevant ones.\n\n"
        + _JSON_FORMAT_INSTRUCTION.format(top_k=top_k)
    )


# ═══════════════════════════════════════════════════════════════════════════
# Response parsing & validation
# ═══════════════════════════════════════════════════════════════════════════

def _parse_llm_response(
    raw: str | None,
    valid_ids: set[str],
    top_k: int,
) -> RouterResult | None:
    """Parse an LLM JSON response into a validated RouterResult.

    Returns ``None`` if:
    - ``raw`` is None or empty
    - JSON parsing fails
    - No returned section_ids exist in the index

    Parameters
    ----------
    raw : str | None
        Raw LLM response text.
    valid_ids : set[str]
        All section_ids that exist in the book's index.
    top_k : int
        Maximum number of section_ids to return.

    Returns
    -------
    RouterResult | None
        Validated result, or None if parsing/validation failed.
    """
    if not raw:
        return None

    # Strip markdown code fences if the LLM wrapped its response
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        # Try to find JSON object within the text
        match = re.search(r"\{[^{}]*\}", cleaned, re.DOTALL)
        if not match:
            return None
        try:
            data = json.loads(match.group())
        except json.JSONDecodeError:
            return None

    # Extract and validate fields
    section_ids = data.get("section_ids", [])
    if isinstance(section_ids, str):
        section_ids = [section_ids]

    # Filter to only valid section IDs that exist in the index
    validated_ids = [sid for sid in section_ids if sid in valid_ids]

    if not validated_ids:
        return None  # All returned IDs were invalid

    # Cap at top_k
    validated_ids = validated_ids[:top_k]

    confidence = data.get("confidence", "low")
    if confidence not in ("high", "low"):
        confidence = "low"

    reasoning = data.get("reasoning", "")
    if not isinstance(reasoning, str):
        reasoning = str(reasoning)

    return RouterResult(
        section_ids=validated_ids,
        confidence=confidence,
        reasoning=reasoning,
    )


# ═══════════════════════════════════════════════════════════════════════════
# TF-IDF keyword fallback (non-LLM)
# ═══════════════════════════════════════════════════════════════════════════

# Minimal stopwords for question tokenization (reuses the concept from
# build_index.py but kept lightweight here to avoid a heavy import).
_QUESTION_STOPWORDS: set[str] = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "must", "can", "could", "what", "which",
    "who", "whom", "when", "where", "why", "how", "in", "on", "at", "to",
    "for", "of", "with", "by", "from", "and", "but", "or", "not", "no",
    "if", "then", "so", "as", "it", "its", "this", "that", "these",
    "those", "i", "me", "my", "we", "you", "your", "he", "she", "they",
    "about", "explain", "describe", "define", "discuss", "between",
}

_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")


def _keyword_fallback(
    question: str,
    index: dict[str, Any],
    top_k: int,
) -> RouterResult:
    """Score sections by keyword overlap with the question (no LLM needed).

    Tokenizes the question, removes stopwords, and computes the
    Jaccard-like overlap with each section's ``tfidf_keywords``.

    This is a cheap insurance fallback — useful when:
    - The LLM call errors out or returns garbage
    - As a baseline to compare the LLM router against in eval

    Parameters
    ----------
    question : str
        The student's question.
    index : dict
        The full book index.
    top_k : int
        Number of sections to return.

    Returns
    -------
    RouterResult
        Always returns ``confidence: "low"`` since this is a fallback.
    """
    # Tokenize the question
    q_tokens = set(
        t for t in _TOKEN_RE.findall(question.lower())
        if t not in _QUESTION_STOPWORDS and len(t) >= 3 and not t.isdigit()
    )

    if not q_tokens:
        return RouterResult(
            section_ids=[],
            confidence="low",
            reasoning="keyword fallback — no meaningful tokens in question",
        )

    # Score each section by overlap
    all_sections = _collect_all_sections_flat(index)
    scored: list[tuple[str, float, str]] = []

    for sec in all_sections:
        sid = sec.get("section_id", "")
        title = sec.get("title", "")
        keywords = set(sec.get("tfidf_keywords", []))

        # Also include words from the title for matching
        title_tokens = set(
            t for t in _TOKEN_RE.findall(title.lower())
            if len(t) >= 3 and not t.isdigit()
        )
        all_kw = keywords | title_tokens

        if not all_kw:
            continue

        # Overlap score: |intersection| / |question_tokens|
        # (biased towards matching more question terms, not section terms)
        overlap = len(q_tokens & all_kw)
        if overlap > 0:
            score = overlap / len(q_tokens)
            scored.append((sid, score, title))

    if not scored:
        return RouterResult(
            section_ids=[],
            confidence="low",
            reasoning="keyword fallback — no keyword overlap found",
        )

    # Sort by score descending, take top_k
    scored.sort(key=lambda x: x[1], reverse=True)
    top_sections = scored[:top_k]

    section_ids = [s[0] for s in top_sections]
    top_score = top_sections[0][1]
    top_title = top_sections[0][2]

    return RouterResult(
        section_ids=section_ids,
        confidence="low",
        reasoning=(
            f"keyword fallback (LLM unavailable) — "
            f"best match: '{top_title}' (score: {top_score:.2f})"
        ),
    )


# ═══════════════════════════════════════════════════════════════════════════
# Main routing logic
# ═══════════════════════════════════════════════════════════════════════════

def _route_single_stage(
    index: dict[str, Any],
    question: str,
    confusable_pairs: list[dict[str, Any]],
    valid_ids: set[str],
    top_k: int,
) -> RouterResult | None:
    """Route using a single flat prompt (for small books).

    Returns None if both attempts fail → caller should use fallback.
    """
    chapters = index.get("chapters", [])

    # Collect all candidate section IDs for confusable context
    all_candidate_ids = list(valid_ids)
    confusable_ctx = _build_confusable_context(
        all_candidate_ids, confusable_pairs, index
    )

    prompt = _build_flat_prompt(chapters, question, confusable_ctx, top_k)

    # Attempt 1
    raw = call_llm(
        prompt,
        temperature=ROUTER_TEMPERATURE,
        max_tokens=300,
        model=ROUTER_MODEL_NAME,
    )
    result = _parse_llm_response(raw, valid_ids, top_k)
    if result:
        return result

    # Attempt 2 — retry with stricter instructions
    warnings.warn(
        "[ROUTER] First LLM attempt failed parsing — retrying with stricter prompt",
        stacklevel=2,
    )
    raw = call_llm(
        prompt + _RETRY_INSTRUCTION,
        temperature=ROUTER_TEMPERATURE,
        max_tokens=300,
        model=ROUTER_MODEL_NAME,
    )
    return _parse_llm_response(raw, valid_ids, top_k)


def _route_two_stage(
    index: dict[str, Any],
    question: str,
    confusable_pairs: list[dict[str, Any]],
    valid_ids: set[str],
    top_k: int,
) -> RouterResult | None:
    """Route using two stages: chapter pick → section pick.

    Stage 1 sends only top-level chapters (titles + abstracts).
    Stage 2 sends the full subtree of the selected chapters.

    Returns None if both stages fail → caller should use fallback.
    """
    chapters = index.get("chapters", [])

    # ── Stage 1: Chapter pick ──────────────────────────────────────────
    stage1_prompt = _build_chapter_pick_prompt(chapters, question)
    raw = call_llm(
        stage1_prompt,
        temperature=ROUTER_TEMPERATURE,
        max_tokens=200,
        model=ROUTER_MODEL_NAME,
    )

    # For Stage 1, valid IDs are just the top-level chapter IDs
    chapter_ids = {ch.get("section_id", "") for ch in chapters}
    stage1_result = _parse_llm_response(raw, chapter_ids, top_k=2)

    if not stage1_result:
        # Stage 1 failed — try a stricter retry
        warnings.warn(
            "[ROUTER] Stage 1 (chapter pick) failed — retrying",
            stacklevel=2,
        )
        raw = call_llm(
            stage1_prompt + _RETRY_INSTRUCTION,
            temperature=ROUTER_TEMPERATURE,
            max_tokens=200,
            model=ROUTER_MODEL_NAME,
        )
        stage1_result = _parse_llm_response(raw, chapter_ids, top_k=2)

        if not stage1_result:
            # Stage 1 completely failed — can't do Stage 2
            return None

    selected_chapter_ids = stage1_result["section_ids"]

    # ── Stage 2: Section pick ──────────────────────────────────────────
    subtrees = _get_chapter_subtrees(index, selected_chapter_ids)

    # Collect all section IDs within the selected chapters for validation
    stage2_valid: set[str] = set()

    def _collect(nodes: list[dict[str, Any]]) -> None:
        for node in nodes:
            sid = node.get("section_id", "")
            if sid:
                stage2_valid.add(sid)
            if node.get("children"):
                _collect(node["children"])

    for st in subtrees:
        sid = st.get("section_id", "")
        if sid:
            stage2_valid.add(sid)
        if st.get("children"):
            _collect(st["children"])

    # Build confusable context for sections in the candidate chapters
    confusable_ctx = _build_confusable_context(
        list(stage2_valid), confusable_pairs, index
    )

    stage2_prompt = _build_section_pick_prompt(
        subtrees, question, confusable_ctx, top_k
    )

    raw = call_llm(
        stage2_prompt,
        temperature=ROUTER_TEMPERATURE,
        max_tokens=300,
        model=ROUTER_MODEL_NAME,
    )
    result = _parse_llm_response(raw, stage2_valid, top_k)
    if result:
        return result

    # Retry Stage 2
    warnings.warn(
        "[ROUTER] Stage 2 (section pick) failed — retrying",
        stacklevel=2,
    )
    raw = call_llm(
        stage2_prompt + _RETRY_INSTRUCTION,
        temperature=ROUTER_TEMPERATURE,
        max_tokens=300,
        model=ROUTER_MODEL_NAME,
    )
    return _parse_llm_response(raw, stage2_valid, top_k)


# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════

def route(
    book_name: str,
    question: str,
    top_k: int = ROUTER_TOP_K_DEFAULT,
) -> RouterResult:
    """Route a student question to the best-matching section(s) in a book.

    This is the main entry point for Phase 1.  It:
    1. Loads the book's index and confusable pairs
    2. Decides single-stage vs. two-stage based on section count
    3. Calls the LLM with appropriate prompts
    4. Validates the response (retries once on failure)
    5. Falls back to keyword overlap if LLM fails entirely

    Parameters
    ----------
    book_name : str
        The snake_case book directory name under ``books/processed/``.
        Example: ``"fluid_mechanics"``
    question : str
        The student's question.
        Example: ``"What is Bernoulli's equation?"``
    top_k : int
        Maximum number of section IDs to return (default from settings).

    Returns
    -------
    RouterResult
        Always returns a result — falls back to keyword overlap if
        LLM routing fails.

    Raises
    ------
    FileNotFoundError
        If the book hasn't been preprocessed (no ``index.json``).

    Examples
    --------
    >>> result = route("fluid_mechanics", "What is Bernoulli's equation?")
    >>> result["section_ids"]
    ['3.2', '3.2.1']
    >>> result["confidence"]
    'high'
    """
    # 1. Load data
    index = _load_index(book_name)
    confusable_pairs = _load_confusable_pairs(book_name)
    valid_ids = _collect_all_section_ids(index)

    if not valid_ids:
        return RouterResult(
            section_ids=[],
            confidence="low",
            reasoning="empty index — no sections found",
        )

    # 2. Decide routing strategy based on section count
    total_sections = _count_all_sections(index)
    use_two_stage = total_sections > TWO_STAGE_THRESHOLD

    # 3. Try LLM routing
    result: RouterResult | None = None

    if LLM_API_KEY and ROUTER_MODEL_NAME:
        if use_two_stage:
            result = _route_two_stage(
                index, question, confusable_pairs, valid_ids, top_k
            )
        else:
            result = _route_single_stage(
                index, question, confusable_pairs, valid_ids, top_k
            )

    # 4. Fall back to keyword overlap if LLM failed
    if result is None:
        if LLM_API_KEY and ROUTER_MODEL_NAME:
            warnings.warn(
                "[ROUTER] LLM routing failed — falling back to keyword overlap",
                stacklevel=2,
            )
        result = _keyword_fallback(question, index, top_k)

    return result


# ═══════════════════════════════════════════════════════════════════════════
# CLI entry-point (for quick testing)
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python -m core.router <book_name> <question>")
        print('Example: python -m core.router fluid_mechanics "What is Bernoulli\'s equation?"')
        sys.exit(1)

    book = sys.argv[1]
    q = " ".join(sys.argv[2:])

    print(f"[ROUTER] Routing question for book '{book}': {q}")
    r = route(book, q)
    print(f"\nResult:")
    print(f"  section_ids : {r['section_ids']}")
    print(f"  confidence  : {r['confidence']}")
    print(f"  reasoning   : {r['reasoning']}")
