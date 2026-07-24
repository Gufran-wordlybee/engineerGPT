"""Phase 1 evaluation harness: measures router accuracy against hand-labeled questions.

This is the single most important file in Phase 1 — it tells you when to stop
iterating on the router prompt and move to Phase 2.

Usage
-----
    # Evaluate a single book:
    python -m evaluation.eval_router --book ai

    # Evaluate all books with question sets:
    python -m evaluation.eval_router --all

    # Verbose mode (prints full LLM responses):
    python -m evaluation.eval_router --book ai --verbose

Exit criteria (from Plan.md):
    Top-1 accuracy ≥ 90% on 15-20 sample questions per book.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so `core.*` and `config.*` imports work
# when running as `python -m evaluation.eval_router` from the project root.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.router import (
    flatten_index,
    load_book_index,
    load_confusable_map,
    route_query,
)
from core.retrieval import shortlist_candidates
from config.settings import ROUTER_SHORTLIST_N

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
QUESTIONS_DIR = Path(__file__).resolve().parent / "questions"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# QUESTION SET LOADING
# ═══════════════════════════════════════════════════════════════════════════

def load_question_set(book_name: str) -> list[dict]:
    """Load the hand-written question set for a book.

    Args:
        book_name: e.g. "ai", "coa"

    Returns:
        List of dicts, each with keys:
        - question: str
        - expected_sections: list[str] — one or more correct section_ids
        - difficulty: str (optional)
        - type: str (optional — "single-topic", "comparison", etc.)
        - notes: str (optional)

    Raises:
        FileNotFoundError: If no question set exists for this book.
    """
    path = QUESTIONS_DIR / f"{book_name}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No question set found at {path}. "
            f"Create evaluation/questions/{book_name}.json first."
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_available_books() -> list[str]:
    """List all books that have question sets."""
    if not QUESTIONS_DIR.exists():
        return []
    return [p.stem for p in QUESTIONS_DIR.glob("*.json")]


def validate_question_set(book_name: str, questions: list[dict], index: dict) -> list[dict]:
    """Return question labels whose expected section IDs are not routable.

    Keeping this check separate lets you run `--validate-only` without making
    any LLM/API calls, which is useful while hand-editing evaluation sets.
    """
    candidates = flatten_index(index)
    valid_ids = {c["section_id"] for c in candidates}
    invalid: list[dict] = []

    for q in questions:
        missing = [
            sid for sid in q.get("expected_sections", [])
            if sid not in valid_ids
        ]
        if missing:
            invalid.append({
                "question": q.get("question", ""),
                "missing_sections": missing,
            })

    return invalid


def evaluate_shortlist_recall(book_name: str, shortlist_n: int = ROUTER_SHORTLIST_N) -> dict:
    """Evaluate Stage A recall without making any LLM/API calls."""
    print(f"\n{'='*70}")
    print(f"  EVALUATING SHORTLIST RECALL: {book_name}")
    print(f"{'='*70}\n")

    questions = load_question_set(book_name)
    index = load_book_index(book_name)
    candidates = flatten_index(index)
    confusable_map = load_confusable_map(book_name)

    invalid_labels = validate_question_set(book_name, questions, index)
    if invalid_labels:
        print("  ERROR: Question set contains expected section IDs that are not routable:")
        for item in invalid_labels:
            print(f"    - {item['missing_sections']} :: {item['question'][:70]}...")
        raise ValueError(
            f"{book_name} question set has {len(invalid_labels)} invalid labels. "
            "Fix evaluation/questions before running shortlist recall."
        )

    results: list[dict] = []
    for i, q in enumerate(questions, 1):
        expected = set(q.get("expected_sections", []))
        shortlisted = shortlist_candidates(
            question=q["question"],
            candidates=candidates,
            top_n=shortlist_n,
        )
        shortlisted_ids = [candidate["section_id"] for candidate in shortlisted]
        shortlisted_set = set(shortlisted_ids)
        expected_found = expected & shortlisted_set
        exact_match = expected.issubset(shortlisted_set)

        confusable_siblings = sorted({
            sibling
            for sid in expected
            for sibling in confusable_map.get(sid, [])
        })
        missing_confusable_siblings = [
            sibling for sibling in confusable_siblings
            if sibling not in shortlisted_set
        ]

        result = {
            "question": q["question"],
            "expected_sections": sorted(expected),
            "shortlisted_ids": shortlisted_ids,
            "shortlist_hit": bool(expected_found),
            "exact_match": exact_match,
            "missing_expected": sorted(expected - shortlisted_set),
            "confusable_involved": bool(confusable_siblings),
            "missing_confusable_siblings": missing_confusable_siblings,
            "difficulty": q.get("difficulty", ""),
            "type": q.get("type", ""),
        }
        results.append(result)

        status = "PASS" if result["shortlist_hit"] else "MISS"
        print(f"  [{i:2d}/{len(questions)}] {q['question'][:65]}... {status}")
        if not result["shortlist_hit"]:
            print(f"       Missing:  {result['missing_expected']}")
            print(f"       Top ids:   {shortlisted_ids[:5]}")

    total = len(results)
    hit_count = sum(1 for result in results if result["shortlist_hit"])
    exact_count = sum(1 for result in results if result["exact_match"])
    split_confusable = [
        result for result in results
        if result["confusable_involved"] and result["missing_confusable_siblings"]
    ]

    recall = hit_count / total if total else 0
    exact_recall = exact_count / total if total else 0

    print(f"\n{'-'*70}")
    print(f"  SHORTLIST RESULTS: {book_name}")
    print(f"{'-'*70}")
    print(f"  Shortlist size:            {shortlist_n}")
    print(f"  Total questions:           {total}")
    print(f"  Any expected in shortlist: {hit_count}/{total} = {recall:.1%}")
    print(f"  All expected in shortlist: {exact_count}/{total} = {exact_recall:.1%}")
    print(f"  Confusable sibling splits: {len(split_confusable)}")
    print(f"{'-'*70}\n")

    return {
        "book_name": book_name,
        "shortlist_n": shortlist_n,
        "total_questions": total,
        "shortlist_recall": round(recall, 4),
        "exact_shortlist_recall": round(exact_recall, 4),
        "confusable_sibling_splits": split_confusable,
        "results": results,
    }


# ═══════════════════════════════════════════════════════════════════════════
# EVALUATION LOGIC
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_single_question(
    question_entry: dict,
    index: dict,
    book_name: str,
    confusable_map: dict[str, list[str]],
    top_k: int = 3,
) -> dict:
    """Run the router on one question and compare against expected sections.

    Returns a result dict with:
    - question: str
    - expected_sections: list[str]
    - predicted_sections: list[dict] — full router output
    - top1_hit: bool — True if top-1 prediction is in expected_sections
    - top3_hit: bool — True if ANY of top-3 predictions is in expected_sections
    - exact_match: bool — True if ALL expected sections appear in predictions
    - confusable_involved: bool — True if expected section has confusable siblings
    - latency_ms: float — time taken for this query
    """
    question = question_entry["question"]
    expected = set(question_entry["expected_sections"])

    start = time.time()
    predicted = route_query(
        question=question,
        index=index,
        top_k=top_k,
        book_name=book_name,
        confusable_map=confusable_map,
    )
    latency_ms = (time.time() - start) * 1000

    predicted_ids = [p["section_id"] for p in predicted]

    # --- Top-1 accuracy ---
    top1_hit = bool(predicted_ids and predicted_ids[0] in expected)

    # --- Top-3 accuracy ---
    top3_hit = bool(expected & set(predicted_ids[:top_k]))

    # --- Exact match (all expected sections found in predictions) ---
    exact_match = expected.issubset(set(predicted_ids))

    # --- Check if confusable pairs are involved ---
    confusable_involved = any(
        sid in confusable_map for sid in expected
    )

    return {
        "question": question,
        "expected_sections": list(expected),
        "predicted_sections": predicted,
        "predicted_ids": predicted_ids,
        "top1_hit": top1_hit,
        "top3_hit": top3_hit,
        "exact_match": exact_match,
        "confusable_involved": confusable_involved,
        "latency_ms": round(latency_ms, 1),
        "difficulty": question_entry.get("difficulty", ""),
        "type": question_entry.get("type", ""),
    }


def evaluate_book(
    book_name: str,
    top_k: int = 3,
    verbose: bool = False,
) -> dict:
    """Run the full evaluation for one book.

    Returns a summary dict with:
    - book_name: str
    - total_questions: int
    - top1_accuracy: float (0.0 - 1.0)
    - top3_accuracy: float
    - exact_match_accuracy: float
    - avg_latency_ms: float
    - results: list[dict] — per-question results
    - misses: list[dict] — only the questions where top-1 was wrong
    - confusable_misses: list[dict] — misses involving confusable pairs
    """
    print(f"\n{'='*70}")
    print(f"  EVALUATING: {book_name}")
    print(f"{'='*70}\n")

    # Load data
    questions = load_question_set(book_name)
    index = load_book_index(book_name)
    confusable_map = load_confusable_map(book_name)

    invalid_labels = validate_question_set(book_name, questions, index)
    if invalid_labels:
        print("  ERROR: Question set contains expected section IDs that are not routable:")
        for item in invalid_labels:
            print(f"    - {item['missing_sections']} :: {item['question'][:70]}...")
        raise ValueError(
            f"{book_name} question set has {len(invalid_labels)} invalid labels. "
            "Fix evaluation/questions before running router accuracy."
        )

    # Run evaluation
    results: list[dict] = []
    for i, q in enumerate(questions, 1):
        print(f"  [{i:2d}/{len(questions)}] {q['question'][:65]}...", end="", flush=True)

        result = evaluate_single_question(
            question_entry=q,
            index=index,
            book_name=book_name,
            confusable_map=confusable_map,
            top_k=top_k,
        )
        results.append(result)

        # Print inline result
        status = "✅" if result["top1_hit"] else "❌"
        print(f" {status} ({result['latency_ms']:.0f}ms)")

        if verbose or not result["top1_hit"]:
            print(f"       Expected: {result['expected_sections']}")
            print(f"       Got:      {result['predicted_ids']}")
            if result["predicted_sections"]:
                top = result["predicted_sections"][0]
                print(f"       Reason:   {top.get('reason', 'N/A')}")
            print()

    # Compute metrics
    total = len(results)
    top1_correct = sum(1 for r in results if r["top1_hit"])
    top3_correct = sum(1 for r in results if r["top3_hit"])
    exact_correct = sum(1 for r in results if r["exact_match"])
    avg_latency = sum(r["latency_ms"] for r in results) / total if total else 0

    top1_accuracy = top1_correct / total if total else 0
    top3_accuracy = top3_correct / total if total else 0
    exact_match_accuracy = exact_correct / total if total else 0

    # Collect misses
    misses = [r for r in results if not r["top1_hit"]]
    confusable_misses = [r for r in misses if r["confusable_involved"]]

    # Print summary
    print(f"\n{'─'*70}")
    print(f"  RESULTS: {book_name}")
    print(f"{'─'*70}")
    print(f"  Total questions:     {total}")
    print(f"  Top-1 accuracy:      {top1_correct}/{total} = {top1_accuracy:.1%}"
          f"  {'✅ PASS' if top1_accuracy >= 0.9 else '❌ FAIL (need ≥90%)'}")
    print(f"  Top-3 accuracy:      {top3_correct}/{total} = {top3_accuracy:.1%}")
    print(f"  Exact match:         {exact_correct}/{total} = {exact_match_accuracy:.1%}")
    print(f"  Avg latency:         {avg_latency:.0f}ms")
    print(f"  Confusable misses:   {len(confusable_misses)}")
    print(f"{'─'*70}\n")

    summary = {
        "book_name": book_name,
        "total_questions": total,
        "top1_accuracy": round(top1_accuracy, 4),
        "top3_accuracy": round(top3_accuracy, 4),
        "exact_match_accuracy": round(exact_match_accuracy, 4),
        "avg_latency_ms": round(avg_latency, 1),
        "results": results,
        "misses": misses,
        "confusable_misses": confusable_misses,
    }

    return summary


# ═══════════════════════════════════════════════════════════════════════════
# RESULTS OUTPUT
# ═══════════════════════════════════════════════════════════════════════════

def save_results(summary: dict) -> Path:
    """Save the full evaluation results and a separate misses file.

    Creates:
    - evaluation/results/<book>_results.json  — full results
    - evaluation/results/<book>_misses.json   — only misses (for quick review)

    Returns:
        Path to the results file.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    book = summary["book_name"]

    # --- Full results ---
    results_path = RESULTS_DIR / f"{book}_results.json"

    # Strip predicted_sections from results to keep the file manageable
    # (the full router output per question can be verbose)
    slim_results = []
    for r in summary["results"]:
        slim_results.append({
            "question": r["question"],
            "expected_sections": r["expected_sections"],
            "predicted_ids": r["predicted_ids"],
            "top1_hit": r["top1_hit"],
            "top3_hit": r["top3_hit"],
            "exact_match": r["exact_match"],
            "confusable_involved": r["confusable_involved"],
            "latency_ms": r["latency_ms"],
            "difficulty": r["difficulty"],
            "type": r["type"],
        })

    output = {
        "book_name": summary["book_name"],
        "total_questions": summary["total_questions"],
        "top1_accuracy": summary["top1_accuracy"],
        "top3_accuracy": summary["top3_accuracy"],
        "exact_match_accuracy": summary["exact_match_accuracy"],
        "avg_latency_ms": summary["avg_latency_ms"],
        "results": slim_results,
    }

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"  📄 Full results saved to: {results_path}")

    # --- Misses file ---
    if summary["misses"]:
        misses_path = RESULTS_DIR / f"{book}_misses.json"
        slim_misses = []
        for r in summary["misses"]:
            slim_misses.append({
                "question": r["question"],
                "expected_sections": r["expected_sections"],
                "predicted_ids": r["predicted_ids"],
                "confusable_involved": r["confusable_involved"],
                "reasons": [
                    {"section_id": p["section_id"], "reason": p.get("reason", "")}
                    for p in r["predicted_sections"]
                ],
            })

        with open(misses_path, "w", encoding="utf-8") as f:
            json.dump(slim_misses, f, indent=2, ensure_ascii=False)
        print(f"  📄 Misses saved to: {misses_path}")
    else:
        print(f"  🎉 No misses — perfect top-1 accuracy!")

    return results_path


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Phase 1 section router accuracy.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m evaluation.eval_router --book ai
  python -m evaluation.eval_router --book coa --verbose
  python -m evaluation.eval_router --all
  python -m evaluation.eval_router --book coa --validate-shortlist
        """,
    )
    parser.add_argument(
        "--book",
        type=str,
        help="Book name to evaluate (e.g. 'ai', 'coa').",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Evaluate all books that have question sets.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of top candidates the router returns (default: 3).",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed output for every question (not just misses).",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save results to disk.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate question labels against index.json without calling the LLM.",
    )
    parser.add_argument(
        "--validate-shortlist",
        action="store_true",
        help="Evaluate TF-IDF shortlist recall without calling the LLM.",
    )
    parser.add_argument(
        "--shortlist-n",
        type=int,
        default=ROUTER_SHORTLIST_N,
        help=f"Number of Stage A candidates to keep (default: {ROUTER_SHORTLIST_N}).",
    )

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(name)s | %(levelname)s | %(message)s",
    )

    # Determine which books to evaluate
    if args.all:
        books = list_available_books()
        if not books:
            print("No question sets found in evaluation/questions/")
            sys.exit(1)
        print(f"Found question sets for: {', '.join(books)}")
    elif args.book:
        books = [args.book]
    else:
        parser.print_help()
        sys.exit(1)

    # Run evaluations
    all_summaries: list[dict] = []
    for book in books:
        try:
            if args.validate_only:
                questions = load_question_set(book)
                index = load_book_index(book)
                invalid_labels = validate_question_set(book, questions, index)
                if invalid_labels:
                    print(f"\n  ERROR: {book} has invalid expected section IDs:")
                    for item in invalid_labels:
                        print(f"    - {item['missing_sections']} :: {item['question'][:70]}...")
                    continue
                print(f"\n  OK: {book} question labels are all routable ({len(questions)} questions).")
                continue

            if args.validate_shortlist:
                summary = evaluate_shortlist_recall(
                    book_name=book,
                    shortlist_n=args.shortlist_n,
                )
                all_summaries.append(summary)
                continue

            summary = evaluate_book(
                book_name=book,
                top_k=args.top_k,
                verbose=args.verbose,
            )
            all_summaries.append(summary)

            if not args.no_save:
                save_results(summary)

        except FileNotFoundError as e:
            print(f"\n  ❌ Error: {e}")
            continue
        except Exception as e:
            print(f"\n  ❌ Unexpected error evaluating '{book}': {e}")
            logger.exception("Full traceback:")
            continue

    # Print overall summary if multiple books
    if len(all_summaries) > 1:
        print(f"\n{'═'*70}")
        print(f"  OVERALL SUMMARY")
        print(f"{'═'*70}")
        for s in all_summaries:
            status = "✅" if s["top1_accuracy"] >= 0.9 else "❌"
            print(f"  {status} {s['book_name']:10s}  "
                  f"Top-1: {s['top1_accuracy']:.1%}  "
                  f"Top-3: {s['top3_accuracy']:.1%}  "
                  f"({s['total_questions']} questions)")
        print(f"{'═'*70}\n")

    # Exit with error code if any book fails the 90% bar
    if args.validate_only or args.validate_shortlist:
        return

    if any(s["top1_accuracy"] < 0.9 for s in all_summaries):
        sys.exit(1)


if __name__ == "__main__":
    main()
