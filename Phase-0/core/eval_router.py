"""
core.eval_router — Evaluation harness for the section router.

Measures routing accuracy against hand-written test questions.  This is
the automated version of plan.md's exit criterion: "90%+ on 15–20
questions per book."

Usage
-----
::

    # Evaluate a single book
    python -m core.eval_router fluid_mechanics

    # Evaluate all books that have test_questions.json
    python -m core.eval_router

Test question format  (``books/processed/<book>/test_questions.json``)
---------------------------------------------------------------------
::

    [
        {
            "question": "What is Bernoulli's equation?",
            "expected_section": "3.2",
            "tags": ["confusable"]
        },
        ...
    ]

- ``question``: the student question to route
- ``expected_section``: the correct section_id
- ``tags`` (optional): labels like ``"confusable"`` for subset analysis

Output
------
Prints accuracy summary + details for every miss so you can see *why*
the router picked wrong and tune prompts accordingly.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from config.settings import BOOKS_PROCESSED_PATH
from core.router import route, RouterResult


# ═══════════════════════════════════════════════════════════════════════════
# Test question loading
# ═══════════════════════════════════════════════════════════════════════════

def _load_test_questions(book_name: str) -> list[dict[str, Any]]:
    """Load test questions for a book from ``test_questions.json``.

    Raises FileNotFoundError if the file doesn't exist.
    """
    path = Path(BOOKS_PROCESSED_PATH) / book_name / "test_questions.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No test_questions.json for book '{book_name}' at {path}.\n"
            f"Create it manually with 15-20 questions in the format:\n"
            f'[{{"question": "...", "expected_section": "3.2", "tags": ["confusable"]}}]'
        )

    with open(path, "r", encoding="utf-8") as fh:
        questions = json.load(fh)

    if not isinstance(questions, list) or not questions:
        raise ValueError(f"test_questions.json must be a non-empty JSON array")

    return questions


# ═══════════════════════════════════════════════════════════════════════════
# Evaluation logic
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_book(book_name: str, *, verbose: bool = True) -> dict[str, Any]:
    """Run the router on all test questions for a book and compute metrics.

    Parameters
    ----------
    book_name : str
        Snake_case book name (directory under ``books/processed/``).
    verbose : bool
        If True, print detailed output including every miss.

    Returns
    -------
    dict
        Evaluation results with keys:
        - ``total``: number of test questions
        - ``top1_correct``: count of top-1 hits
        - ``topk_correct``: count of top-k hits
        - ``top1_accuracy``: top-1 accuracy as a fraction
        - ``topk_accuracy``: top-k accuracy as a fraction
        - ``confusable_total``: count of confusable-tagged questions
        - ``confusable_correct``: count of confusable questions answered correctly
        - ``misses``: list of missed questions with details
    """
    questions = _load_test_questions(book_name)
    total = len(questions)

    top1_correct = 0
    topk_correct = 0
    confusable_total = 0
    confusable_correct = 0
    misses: list[dict[str, Any]] = []

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"  Evaluating router for: {book_name}")
        print(f"  Questions: {total}")
        print(f"{'=' * 60}\n")

    for i, q in enumerate(questions, 1):
        question = q["question"]
        expected = q["expected_section"]
        tags = q.get("tags", [])
        is_confusable = "confusable" in tags

        # Route the question
        result: RouterResult = route(book_name, question)
        returned_ids = result["section_ids"]
        confidence = result["confidence"]
        reasoning = result["reasoning"]

        # Check top-1
        is_top1 = len(returned_ids) > 0 and returned_ids[0] == expected
        # Check top-k
        is_topk = expected in returned_ids

        if is_top1:
            top1_correct += 1
        if is_topk:
            topk_correct += 1

        if is_confusable:
            confusable_total += 1
            if is_top1:
                confusable_correct += 1

        # Track misses
        status = "OK" if is_top1 else ("~" if is_topk else "MISS")

        if verbose:
            print(f"  [{status}] Q{i}: {question[:80]}")
            if not is_top1:
                print(f"       Expected : {expected}")
                print(f"       Got      : {returned_ids}")
                print(f"       Conf     : {confidence}")
                print(f"       Reason   : {reasoning}")
                if is_confusable:
                    print(f"       Tags     : CONFUSABLE")
                print()

        if not is_top1:
            misses.append({
                "question_index": i,
                "question": question,
                "expected": expected,
                "got": returned_ids,
                "confidence": confidence,
                "reasoning": reasoning,
                "tags": tags,
                "in_topk": is_topk,
            })

    # Compute metrics
    top1_acc = top1_correct / total if total > 0 else 0
    topk_acc = topk_correct / total if total > 0 else 0
    conf_acc = (
        confusable_correct / confusable_total
        if confusable_total > 0
        else float("nan")
    )

    results = {
        "book_name": book_name,
        "total": total,
        "top1_correct": top1_correct,
        "topk_correct": topk_correct,
        "top1_accuracy": top1_acc,
        "topk_accuracy": topk_acc,
        "confusable_total": confusable_total,
        "confusable_correct": confusable_correct,
        "confusable_accuracy": conf_acc,
        "misses": misses,
    }

    if verbose:
        _print_summary(results)

    return results


def _print_summary(results: dict[str, Any]) -> None:
    """Print a formatted summary of evaluation results."""
    total = results["total"]
    top1_acc = results["top1_accuracy"]
    topk_acc = results["topk_accuracy"]
    conf_total = results["confusable_total"]
    conf_acc = results.get("confusable_accuracy", float("nan"))
    misses = results["misses"]
    book = results["book_name"]

    print(f"\n{'-' * 60}")
    print(f"  RESULTS: {book}")
    print(f"{'-' * 60}")
    print(f"  Top-1 accuracy : {results['top1_correct']}/{total}  ({top1_acc:.0%})")
    print(f"  Top-k accuracy : {results['topk_correct']}/{total}  ({topk_acc:.0%})")

    if conf_total > 0:
        print(
            f"  Confusable acc : {results['confusable_correct']}/{conf_total}  "
            f"({conf_acc:.0%})"
        )
    else:
        print(f"  Confusable acc : (no confusable questions)")

    # Pass/fail verdict
    target = 0.90
    verdict = "PASS" if top1_acc >= target else "FAIL"
    print(f"\n  Target: >={target:.0%} top-1 -> {verdict}")

    if misses:
        print(f"\n  Misses ({len(misses)}):")
        for m in misses:
            topk_note = " (in top-k)" if m["in_topk"] else ""
            print(f"    Q{m['question_index']}: {m['question'][:60]}")
            print(f"      expected={m['expected']}, got={m['got']}{topk_note}")

    print(f"{'-' * 60}\n")


# ═══════════════════════════════════════════════════════════════════════════
# CLI entry-point
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    """Evaluate one or all books from the command line.

    Usage::

        python -m core.eval_router              # all books with test_questions.json
        python -m core.eval_router fluid_mechanics   # single book
    """
    args = sys.argv[1:]

    if args:
        # Evaluate a specific book
        book_name = args[0]
        try:
            evaluate_book(book_name)
        except FileNotFoundError as exc:
            print(f"[EVAL] Error: {exc}")
            sys.exit(1)
    else:
        # Discover all books that have test_questions.json
        processed = Path(BOOKS_PROCESSED_PATH)
        if not processed.exists():
            print(f"[EVAL] No processed books directory: {processed}")
            sys.exit(1)

        books_found = 0
        for book_dir in sorted(processed.iterdir()):
            tq_path = book_dir / "test_questions.json"
            if book_dir.is_dir() and tq_path.exists():
                books_found += 1
                evaluate_book(book_dir.name)

        if books_found == 0:
            print(
                "[EVAL] No books with test_questions.json found.\n"
                "Create one at books/processed/<book>/test_questions.json"
            )
            sys.exit(1)


if __name__ == "__main__":
    main()
