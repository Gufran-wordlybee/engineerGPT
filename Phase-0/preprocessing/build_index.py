"""
build_index.py — Generate a **hierarchical**, TF-IDF-weighted index for the router.

Upgrades over the original flat bag-of-nouns index:

1. **Tree structure** — chapters contain nested children, matching the book's
   actual hierarchy.  Enables two-stage routing (chapter → section).
2. **TF-IDF keywords** — corpus-aware distinctiveness so common words like
   "algorithm" get down-weighted while locally unique terms get boosted.
   (niche smjhaya hai same chiz cmts me)
3. **LLM-generated abstracts** (optional) — rich semantic descriptions for
   each section, generated once at build time.  Falls back to first-2-sentence
   extraction when disabled.
   (niche smjhaya hai same chiz cmts me, if llm good else 2 lines fallback)
4. **Confusable-pair detection** — flags sibling sections with high keyword
   overlap so you can verify the split quality.
5. **has_visuals flag** — lets the router know which sections have diagrams
   or equations relevant to visual questions.

Public API
----------
build_index(book_name, output_dir) -> dict

build_index.py converts hundreds of section JSON files into a single smart, hierarchical index.json that the router later uses to decide which chapter/section should answer the user's question.
"""

from __future__ import annotations

import json
import math
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

from config.settings import (
    TOP_N_KEYWORDS,
    LLM_ABSTRACTS_ENABLED,
    LLM_API_KEY,
    LLM_MODEL_NAME,
)
# ═══════════════════════════════════════════════════════════════════════════
# Step 1 - raw_sections = _load_sections(sections_dir)
# this is the main function that reads/loads the sections like sect1,sect2 and so on and builds the index.json file
def _load_sections(sections_dir: Path) -> list[dict[str, Any]]:
    """Load all section JSONs, sorted by start_page."""
    sections: list[dict[str, Any]] = []
    if not sections_dir.is_dir():
        print(f"[INDEX] Warning: sections directory not found — {sections_dir}")
        return sections

    for json_path in sorted(sections_dir.glob("*.json")):
        with open(json_path, "r", encoding="utf-8") as fh:
            sections.append(json.load(fh))

    sections.sort(key=lambda s: s.get("start_page", 0))
    return sections
# ═══════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════
# Stopwords
# ═══════════════════════════════════════════════════════════════════════════

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
    "one", "two", "first", "new", "like", "time", "way",
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
    "typically", "approximately", "corresponding",
}

_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")


# ═══════════════════════════════════════════════════════════════════════════
# Step 2 - tfidf_map = _compute_tfidf_keywords(...)
# Tokenization & keywords
# _compute_tfidf_keywords : we have sect1, sect2, sect3, sect4, sect5, sect6, sect7, sect8, sect9, sect10 and so on 
# sect1 has keywords: [a, b, c, d, e], sect2 has keywords: [c, d, e, f, g] so what it doesTF-IDF says sect1 and sect2 have 3 keywords in common, so a,b, f,g are unique keywords for sect1 and sect2 respectively. So it will return a dict with sect1: [a,b] and sect2: [f,g] as unique keywords.


def _tokenize(text: str) -> list[str]:
    """Split on whitespace/punctuation and lowercase."""
    return _TOKEN_RE.findall(text.lower())


def _filter_tokens(tokens: list[str]) -> list[str]:
    """Remove stopwords, short tokens, and pure numbers."""
    return [
        t for t in tokens
        if t not in STOPWORDS and len(t) >= 3 and not t.isdigit()
    ]


def _extract_tf_keywords(text: str, top_n: int) -> list[str]:
    """Return top_n keywords by raw term frequency."""
    filtered = _filter_tokens(_tokenize(text))
    return [w for w, _ in Counter(filtered).most_common(top_n)]


def _compute_tfidf_keywords(
    sections: list[dict[str, Any]],
    top_n: int,
) -> dict[str, list[str]]:
    """Compute TF-IDF keywords for each section across the whole book.

    Returns {section_id: [keyword, ...]}.
    """
    total_docs = len(sections)
    if total_docs == 0:
        return {}

    # Step 1: compute per-section filtered token lists
    section_tokens: dict[str, list[str]] = {}
    for sec in sections:
        sid = sec.get("section_id", "")
        text = sec.get("text", "")
        section_tokens[sid] = _filter_tokens(_tokenize(text))

    # Step 2: document frequency — in how many sections does each term appear?
    doc_freq: Counter[str] = Counter()
    for sid, tokens in section_tokens.items():
        unique = set(tokens)
        for t in unique:
            doc_freq[t] += 1

    # Step 3: TF-IDF per section
    result: dict[str, list[str]] = {}
    for sec in sections:
        sid = sec.get("section_id", "")
        tokens = section_tokens[sid]
        if not tokens:
            result[sid] = []
            continue

        tf = Counter(tokens)
        tfidf_scores: dict[str, float] = {}
        for term, count in tf.items():
            df = doc_freq.get(term, 1)
            idf = math.log(total_docs / df) if df > 0 else 0
            tfidf_scores[term] = count * idf

        sorted_terms = sorted(tfidf_scores, key=tfidf_scores.get, reverse=True)
        result[sid] = sorted_terms[:top_n]

    return result
# ═══════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════
# Step 3 - _generate_abstract()
# Abstracts
# Makes a short summary.Instead of storing 15 pages of text it stores "This section explains Binary Search on sorted arrays..." Router can understand the section faster.If LLM is disabled -> First two sentences become summary.

def _text_based_abstract(text: str) -> str:
    """Extract the first 2 meaningful sentences as a fallback abstract."""
    # Split on sentence-ending punctuation
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    # Filter out very short fragments
    meaningful = [s for s in sentences if len(s) > 20]
    return " ".join(meaningful[:2]) if meaningful else text[:300]


def _call_llm(prompt: str) -> str | None:
    """Call the configured LLM.  Returns response text or None on failure.

    Auto-detects provider from LLM_MODEL_NAME.
    """
    if not LLM_API_KEY or not LLM_MODEL_NAME:
        return None

    model = LLM_MODEL_NAME.lower()

    # Try OpenAI-compatible API (covers OpenAI, most providers)
    if "claude" not in model:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=LLM_API_KEY)
            resp = client.chat.completions.create(
                model=LLM_MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=500,
            )
            return resp.choices[0].message.content
        except ImportError:
            pass
        except Exception as exc:
            warnings.warn(f"[INDEX] OpenAI call failed: {exc}", stacklevel=2)
            return None

    # Try Anthropic API
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=LLM_API_KEY)
        resp = client.messages.create(
            model=LLM_MODEL_NAME,
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text
    except ImportError:
        warnings.warn(
            "[INDEX] Neither 'openai' nor 'anthropic' package installed. "
            "Falling back to text-based abstracts.",
            stacklevel=2,
        )
        return None
    except Exception as exc:
        warnings.warn(f"[INDEX] Anthropic call failed: {exc}", stacklevel=2)
        return None


import warnings


def _generate_abstract(section: dict[str, Any]) -> str:
    """Generate a section abstract — LLM if enabled, else text-based."""
    text = section.get("text", "")

    if not LLM_ABSTRACTS_ENABLED or not LLM_API_KEY:
        return _text_based_abstract(text)

    prompt = (
        "You are summarizing a section of an engineering textbook for a "
        "study assistant.\n"
        f"Section title: {section.get('title', '')}\n"
        f"Chapter: {section.get('chapter', '')}\n\n"
        f"Section content (first 2000 chars):\n{text[:2000]}\n\n"
        "Provide:\n"
        "1. A 2-3 sentence description of what this section covers.\n"
        "2. A list of 5-10 key named concepts/terms.\n"
        "3. Two example question phrasings a student might ask that this "
        "section would answer.\n\n"
        "Format as JSON: "
        '{\"description\": \"...\", \"concepts\": [...], \"example_questions\": [...]}'
    )

    result = _call_llm(prompt)
    if result:
        # Try to extract description from JSON response
        try:
            parsed = json.loads(result)
            desc = parsed.get("description", "")
            concepts = parsed.get("concepts", [])
            questions = parsed.get("example_questions", [])
            parts = [desc]
            if concepts:
                parts.append("Key concepts: " + ", ".join(concepts[:10]))
            if questions:
                parts.append("Example questions: " + "; ".join(questions[:2]))
            return " | ".join(parts)
        except json.JSONDecodeError:
            return result   # Use raw LLM response

    return _text_based_abstract(text)
# ═══════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════
# Step 4 - _build_tree()
# TREE BUILDING
"""
Suppose sections are
Chapter 1

   1.1
   1.2
      1.2.1
      1.2.2
Instead of storing (what we are calling flat againngain)
1
1.1
1.2
1.2.1
it creates 
1
├──1.1
└──1.2
      ├──1.2.1
      └──1.2.2
"""
""" {
   "title":"Binary Search",
   "keywords":[...],
   "tfidf_keywords":[...],
   "abstract":"...",
   "has_visuals":true,
   "children":[...]
}"""
# Notice, No actual textbook text is stored. Only metadata. 

def _build_tree(
    sections: list[dict[str, Any]],
    tfidf_map: dict[str, list[str]],
) -> list[dict[str, Any]]:
    """Build a nested tree from flat section list using parent_id/children.

    Returns the top-level chapters (sections with parent_id == None).
    """
    # Build lookup
    id_to_section: dict[str, dict[str, Any]] = {}
    for sec in sections:
        sid = sec.get("section_id", "")
        text = sec.get("text", "")
        start_page = sec.get("start_page", 0)
        end_page = sec.get("end_page", start_page)
        page_range = (
            f"{start_page + 1}-{end_page + 1}"
            if start_page != end_page
            else str(start_page + 1)
        )

        has_visuals = bool(sec.get("images") or sec.get("equations"))

        node: dict[str, Any] = {
            "section_id": sid,
            "title": sec.get("title", ""),
            "level": sec.get("level", 1),
            "page_range": page_range,
            "keywords": _extract_tf_keywords(text, TOP_N_KEYWORDS),
            "tfidf_keywords": tfidf_map.get(sid, []),
            "abstract": _generate_abstract(sec),
            "has_visuals": has_visuals,
            "children": [],   # populated below
        }
        id_to_section[sid] = node

    # Wire up parent-child
    roots: list[dict[str, Any]] = []
    for sec in sections:
        sid = sec.get("section_id", "")
        parent_id = sec.get("parent_id")
        node = id_to_section[sid]

        if parent_id and parent_id in id_to_section:
            id_to_section[parent_id]["children"].append(node)
        else:
            roots.append(node)

    return roots
# ═══════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════
#Step 5 - Confusable pair detection

def _detect_confusable_pairs(
    index: dict[str, Any],
) -> list[dict[str, Any]]:
    """Find sibling section pairs with > 60% keyword overlap."""
    confusable: list[dict[str, Any]] = []

    def _check_children(children: list[dict[str, Any]]) -> None:
        """Check all pairs among a list of sibling sections."""
        for i in range(len(children)):
            kw_a = set(children[i].get("keywords", []) +
                       children[i].get("tfidf_keywords", []))
            for j in range(i + 1, len(children)):
                kw_b = set(children[j].get("keywords", []) +
                           children[j].get("tfidf_keywords", []))
                if not kw_a or not kw_b:
                    continue
                shared = kw_a & kw_b
                overlap = len(shared) / max(len(kw_a), len(kw_b))
                if overlap > 0.6:
                    confusable.append({
                        "section_a": children[i]["section_id"],
                        "section_b": children[j]["section_id"],
                        "overlap": round(overlap, 2),
                        "shared_keywords": sorted(shared),
                    })

            # Recurse into grandchildren
            if children[i].get("children"):
                _check_children(children[i]["children"])

    for chapter in index.get("chapters", []):
        if chapter.get("children"):
            _check_children(chapter["children"])
        # Also check chapter-level siblings
    _check_children(index.get("chapters", []))

    return confusable
# ═══════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════

def build_index(book_name: str, output_dir: str) -> dict[str, Any]:
    """Build hierarchical, TF-IDF-weighted index with optional LLM abstracts.

    Parameters
    ----------
    book_name : str
        Human-readable book name.
    output_dir : str
        Base output directory for this book.

    Returns
    -------
    dict
        The complete index (also saved to ``<output_dir>/index.json``).
    """
    output_path = Path(output_dir)
    sections_dir = output_path / "sections"

    # 1. Load all sections
    raw_sections = _load_sections(sections_dir)
    total = len(raw_sections)

    # 2. Compute TF-IDF keywords
    print(f"[INDEX] Computing TF-IDF keywords across {total} sections...")
    tfidf_map = _compute_tfidf_keywords(raw_sections, TOP_N_KEYWORDS)

    # 3. Log abstract mode
    if LLM_ABSTRACTS_ENABLED and LLM_API_KEY:
        print(f"[INDEX] Generating LLM abstracts (model: {LLM_MODEL_NAME})...")
    else:
        print("[INDEX] LLM abstracts disabled — using text-based abstracts")

    # 4. Build hierarchical tree
    chapters = _build_tree(raw_sections, tfidf_map)

    # 5. Assemble index
    index: dict[str, Any] = {
        "book_name": book_name,
        "total_sections": total,
        "chapters": chapters,
    }

    # 6. Confusable pair detection
    confusable = _detect_confusable_pairs(index)
    if confusable:
        conf_path = output_path / "confusable_pairs.json"
        with open(conf_path, "w", encoding="utf-8") as fh:
            json.dump(confusable, fh, indent=2, ensure_ascii=False)
        print(
            f"[INDEX] Confusable pairs: {len(confusable)} pairs "
            f"with >60% keyword overlap — see confusable_pairs.json"
        )
    else:
        print("[INDEX] Confusable pairs: none detected")

    # 7. Write index
    index_path = output_path / "index.json"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with open(index_path, "w", encoding="utf-8") as fh:
        json.dump(index, fh, indent=2, ensure_ascii=False)

    print(
        f"[INDEX] Built index for '{book_name}': "
        f"{total} sections (hierarchical), saved to index.json"
    )

    return index


# ═══════════════════════════════════════════════════════════════════════════
# CLI entry-point
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python -m preprocessing.build_index <book_name> <output_dir>")
        sys.exit(1)

    build_index(sys.argv[1], sys.argv[2])
