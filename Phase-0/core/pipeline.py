"""One callable query flow shared by future CLI and Streamlit interfaces."""

from config.settings import ROUTER_TOP_K
from core.generator import generate_answer
from core.router import load_book_index, route_query


def run_query(book_name: str, question: str, top_k: int = ROUTER_TOP_K) -> dict:
    """Route a question, then generate a grounded answer from those sections."""
    result = {
        "question": question,
        "book_name": book_name,
        "routed_sections": [],
        "answer": "",
        "images_used": [],
        "used_vision": False,
    }
    try:
        routed = route_query(question, load_book_index(book_name), top_k, book_name)
    except (FileNotFoundError, ValueError) as exc:
        result["answer"] = f"I could not open the processed book: {exc}"
        return result
    if not routed:
        result["answer"] = "I could not find a relevant textbook section for that question."
        return result
    generated = generate_answer(question, book_name, [item["section_id"] for item in routed])
    result.update(generated)
    result["routed_sections"] = routed
    return result
