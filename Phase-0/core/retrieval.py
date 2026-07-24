"""Stage A: local, non-LLM candidate shortlisting via TF-IDF.

The router should never ask an LLM to search the whole book catalog in one
prompt. This module keeps that prompt small by doing cheap lexical retrieval
locally, then passing only a compact shortlist to the LLM for final ranking.
"""

from __future__ import annotations

import logging
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


def _candidate_text(candidate: dict) -> str:
    """Build the text blob used for TF-IDF similarity for one candidate."""
    title = candidate.get("title", "")
    chapter = candidate.get("chapter_title", "")
    keywords = " ".join(candidate.get("tfidf_keywords", []))
    abstract = candidate.get("abstract", "")

    # Repeat high-signal fields to bias scoring toward direct title/chapter
    # and keyword matches without introducing a custom weighting scheme.
    return f"{title} {title} {chapter} {keywords} {keywords} {abstract}"


def shortlist_candidates(
    question: str,
    candidates: list[dict],
    top_n: int = 20,
) -> list[dict]:
    """Return candidates ordered by descending TF-IDF cosine similarity.

    Args:
        question: Student's natural-language query.
        candidates: Full flattened candidate list from ``flatten_index()``.
        top_n: Number of candidates to keep for LLM re-ranking.

    Returns:
        Up to ``top_n`` candidates. If the candidate list is already small, it
        is returned unchanged. If TF-IDF cannot build a meaningful vocabulary,
        this falls back to token overlap and logs a warning.
    """
    if top_n <= 0:
        logger.warning("Invalid shortlist size %d; keeping all candidates.", top_n)
        return candidates

    if len(candidates) <= top_n:
        return candidates

    corpus = [_candidate_text(candidate) for candidate in candidates]
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=20000,
    )

    try:
        doc_vectors = vectorizer.fit_transform(corpus)
        query_vector = vectorizer.transform([question])
    except ValueError as exc:
        logger.warning(
            "TF-IDF shortlisting failed (%s); falling back to token overlap.",
            exc,
        )
        query_terms = set(re.findall(r"[a-z0-9]+", question.lower()))
        ranked = sorted(
            candidates,
            key=lambda candidate: len(
                query_terms
                & set(re.findall(r"[a-z0-9]+", _candidate_text(candidate).lower()))
            ),
            reverse=True,
        )
        return ranked[:top_n]

    scores = cosine_similarity(query_vector, doc_vectors)[0]
    ranked_idx = scores.argsort()[::-1][:top_n]
    shortlisted = [candidates[i] for i in ranked_idx]

    logger.debug(
        "TF-IDF shortlist: %d/%d candidates kept, top score=%.3f, cutoff score=%.3f",
        len(shortlisted),
        len(candidates),
        scores[ranked_idx[0]],
        scores[ranked_idx[-1]],
    )
    return shortlisted


if __name__ == "__main__":
    demo_candidates = [
        {"section_id": "a", "title": "and", "chapter_title": "", "tfidf_keywords": [], "abstract": ""},
        {"section_id": "b", "title": "or", "chapter_title": "", "tfidf_keywords": [], "abstract": ""},
    ]
    assert shortlist_candidates("or", demo_candidates, top_n=1)[0]["section_id"] == "b"
