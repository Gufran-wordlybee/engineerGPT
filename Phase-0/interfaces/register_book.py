"""Register one already-processed book so it appears in the web UI."""

from __future__ import annotations

import argparse

from interfaces.db import add_book
from interfaces.ui_sidebar import has_book_content


def main() -> None:
    parser = argparse.ArgumentParser(description="Register a processed EngineerGPT book.")
    parser.add_argument("book_id", help="Exact folder name under books/processed/")
    parser.add_argument("display_name", help="Name shown in the web sidebar")
    args = parser.parse_args()
    if not has_book_content(args.book_id):
        parser.error(f"books/processed/{args.book_id}/index.json does not exist")
    add_book(args.book_id, args.display_name)
    print(f"Registered {args.book_id!r}. Push its processed folder, then redeploy Streamlit.")


if __name__ == "__main__":
    main()
