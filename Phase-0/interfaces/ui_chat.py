"""Display helpers kept separate from the Streamlit page flow."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from config.settings import BOOKS_PROCESSED_PATH
from core.generator import load_section


def format_citation_footer(book_id: str, section_ids: list[str]) -> str:
    """Build citations from the processed JSON; no duplicated citation database."""
    lines = []
    for section_id in section_ids:
        try:
            section = load_section(book_id, section_id)
        except (FileNotFoundError, ValueError):
            continue  # An old answer should still render after a book update.
        pages = f"p. {section['start_page']}" if section["start_page"] == section["end_page"] else (
            f"pp. {section['start_page']}–{section['end_page']}"
        )
        lines.append(f"- §{section['title']} ({pages}) · id: `{section['section_id']}`")
    return "---\n📖 **Sources:**\n" + "\n".join(lines) if lines else ""


def answer_with_sources(book_id: str, answer: str, section_ids: list[str]) -> str:
    footer = format_citation_footer(book_id, section_ids)
    return f"{answer}\n\n{footer}" if footer else answer


def render_images(book_id: str, image_paths: list[str], caption: str = "Source image") -> None:
    """Render only files inside this book, even if a stored DB row is malformed."""
    root = (BOOKS_PROCESSED_PATH / book_id).resolve()
    for relative_path in image_paths or []:
        path = (root / relative_path).resolve()
        if path.is_file() and root in path.parents:
            st.image(str(path), caption=caption)


def render_messages(book_id: str, messages: list[dict]) -> None:
    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                render_images(book_id, message.get("images_used") or [])


if __name__ == "__main__":
    assert "p. 7" in format_citation_footer("ai", ["1-1-what-is-ai"])
