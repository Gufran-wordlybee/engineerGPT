"""Small Supabase data-access layer for books, chats, and messages."""

from __future__ import annotations

import os
from datetime import datetime, timezone


def _client():
    """Create a client only when it is needed, so imports stay lightweight."""
    url = os.getenv("SUPABASE_URL")
    # The Streamlit server keeps this key private.  Prefer the service key so
    # the database can keep RLS enabled without exposing public table access.
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_ANON_KEY")
    if not url or not key:
        raise RuntimeError("Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY before using chat history.")
    from supabase import create_client

    return create_client(url, key)


def list_books() -> list[dict]:
    return _client().table("books").select("*").order("display_name").execute().data


def add_book(book_id: str, display_name: str) -> dict:
    """Register or update a deployable processed-book folder."""
    return _client().table("books").upsert({
        "book_id": book_id,
        "display_name": display_name,
    }).execute().data[0]


def list_threads(book_id: str) -> list[dict]:
    return _client().table("threads").select("*").eq("book_id", book_id).order(
        "updated_at", desc=True
    ).execute().data


def create_thread(book_id: str) -> dict:
    return _client().table("threads").insert({"book_id": book_id}).execute().data[0]


def rename_thread(thread_id: str, title: str) -> None:
    _client().table("threads").update({"title": title}).eq("thread_id", thread_id).execute()


def list_messages(thread_id: str) -> list[dict]:
    return _client().table("messages").select("*").eq("thread_id", thread_id).order(
        "created_at"
    ).execute().data


def add_message(
    thread_id: str,
    role: str,
    content: str,
    routed_sections: list[dict] | None = None,
    images_used: list[str] | None = None,
) -> dict:
    """Persist exactly what the user saw, plus answer debugging metadata."""
    message = _client().table("messages").insert({
        "thread_id": thread_id,
        "role": role,
        "content": content,
        "routed_sections": routed_sections,
        "images_used": images_used,
    }).execute().data[0]
    # The sidebar orders chats by activity, not by their original creation date.
    _client().table("threads").update({
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }).eq("thread_id", thread_id).execute()
    return message
