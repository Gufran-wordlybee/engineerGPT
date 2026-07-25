"""EngineerGPT's deployed Streamlit chat interface."""

from __future__ import annotations

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import streamlit as st
# Streamlit Cloud exposes secrets through st.secrets, while the existing core
# reads environment variables. Copying them here keeps the backend unchanged.
try:
    secrets = st.secrets.items()
except FileNotFoundError:  # Local first run: show a useful database error below.
    secrets = []
for key, value in secrets:
    os.environ.setdefault(key, str(value))
from core.pipeline import run_query  # noqa: E402 - secrets must be loaded first
from interfaces import db  # noqa: E402
from interfaces.ui_chat import answer_with_sources, render_messages  # noqa: E402
from interfaces.ui_sidebar import has_book_content  # noqa: E402


st.set_page_config(
    page_title="EngineerGPT",
    page_icon=Path(__file__).resolve().parents[1] / "imgs" / "engineerGPT.png",
    layout="wide",
)
st.title("EngineerGPT")


def short_title(question: str) -> str:
    """Title a new chat locally; spending an LLM call here adds no value."""
    words = question.split()
    return " ".join(words[:6]) + ("…" if len(words) > 6 else "")


def select_book(books: list[dict]) -> dict | None:
    """Keep selection state across Streamlit's normal reruns."""
    if not books:
        st.sidebar.info("No books are registered yet. Run register_book.py first.")
        return None
    valid_ids = {book["book_id"] for book in books}
    if st.session_state.get("book_id") not in valid_ids:
        st.session_state.book_id = books[0]["book_id"]
        st.session_state.pop("thread_id", None)
    selected_id = st.sidebar.radio(
        "Books",
        options=[book["book_id"] for book in books],
        index=[book["book_id"] for book in books].index(st.session_state.book_id),
        format_func=lambda book_id: next(book["display_name"] for book in books if book["book_id"] == book_id),
    )
    if selected_id != st.session_state.book_id:
        st.session_state.book_id = selected_id
        st.session_state.pop("thread_id", None)
        st.rerun()
    return next(book for book in books if book["book_id"] == selected_id)


def select_thread(book_id: str) -> dict | None:
    threads = db.list_threads(book_id)
    st.sidebar.divider()
    if st.sidebar.button("+ New chat", use_container_width=True):
        st.session_state.thread_id = db.create_thread(book_id)["thread_id"]
        st.rerun()
    if not threads:
        return None
    thread_ids = {thread["thread_id"] for thread in threads}
    if st.session_state.get("thread_id") not in thread_ids:
        st.session_state.thread_id = threads[0]["thread_id"]
    st.sidebar.caption("Chats")
    chosen_id = st.sidebar.radio(
        "Chats",
        options=[thread["thread_id"] for thread in threads],
        index=[thread["thread_id"] for thread in threads].index(st.session_state.thread_id),
        format_func=lambda thread_id: next(thread["title"] for thread in threads if thread["thread_id"] == thread_id),
        label_visibility="collapsed",
    )
    if chosen_id != st.session_state.thread_id:
        st.session_state.pop("delete_thread_id", None)
    st.session_state.thread_id = chosen_id
    thread = next(thread for thread in threads if thread["thread_id"] == chosen_id)

    with st.sidebar.form("rename_chat"):
        title = st.text_input("Rename chat", value=thread["title"], key=f"rename_chat_{chosen_id}")
        if st.form_submit_button("Rename", use_container_width=True):
            if title := title.strip():
                db.rename_thread(chosen_id, title)
                st.rerun()
            st.warning("Enter a chat title.")

    if st.sidebar.button("Delete chat", use_container_width=True):
        st.session_state.delete_thread_id = chosen_id
    if st.session_state.get("delete_thread_id") == chosen_id:
        st.sidebar.warning("Delete this chat permanently?")
        confirm, cancel = st.sidebar.columns(2)
        if confirm.button("Delete", type="primary", use_container_width=True):
            db.delete_thread(chosen_id)
            st.session_state.pop("thread_id", None)
            st.session_state.pop("delete_thread_id", None)
            st.rerun()
        if cancel.button("Cancel", use_container_width=True):
            st.session_state.pop("delete_thread_id", None)
            st.rerun()
    return thread


try:
    selected_book = select_book(db.list_books())
    if not selected_book:
        st.stop()
    if not has_book_content(selected_book["book_id"]):
        st.error("This book is registered but its processed folder is missing from this deployment.")
        st.stop()
    selected_thread = select_thread(selected_book["book_id"])
except RuntimeError as exc:
    st.error(str(exc))
    st.stop()

if not selected_thread:
    st.info("Choose **+ New chat** to start asking questions about this book.")
    st.stop()

st.subheader(selected_thread["title"])
messages = db.list_messages(selected_thread["thread_id"])
render_messages(selected_book["book_id"], messages)

if question := st.chat_input("Ask a question about this textbook"):
    # Save first: a browser refresh cannot lose the student's question.
    db.add_message(selected_thread["thread_id"], "user", question)
    if not messages:
        db.rename_thread(selected_thread["thread_id"], short_title(question))
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Searching the textbook…"):
            result = run_query(selected_book["book_id"], question)
            answer = answer_with_sources(
                selected_book["book_id"], result["answer"], result.get("sections_used", [])
            )
            db.add_message(
                selected_thread["thread_id"],
                "assistant",
                answer,
                result.get("routed_sections"),
                result.get("images_used"),
            )
        st.markdown(answer)
        from interfaces.ui_chat import render_images
        render_images(selected_book["book_id"], result.get("images_used", []))
