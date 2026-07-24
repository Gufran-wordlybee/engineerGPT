"""Minimal live smoke test for Phase 2; run with ``python -m core.tester_gen``."""

from core.pipeline import run_query


if __name__ == "__main__":
    result = run_query("coa", "difference between parallel and priority interrupts")
    print(result["answer"])
    print("Sections used:", [item["section_id"] for item in result["routed_sections"]])
    print("Images used:", result["images_used"])
