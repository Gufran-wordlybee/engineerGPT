"""Sidebar navigation for the selected book and its persisted chat threads."""

from __future__ import annotations

from config.settings import BOOKS_PROCESSED_PATH


def has_book_content(book_id: str) -> bool:
    """A registry row without its deployed folder must not crash the app."""
    return (BOOKS_PROCESSED_PATH / book_id / "index.json").is_file()
