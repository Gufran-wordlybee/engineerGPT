"""
Core pipeline: ties router and generator together into a single callable flow.

Both CLI and Streamlit interfaces call this module, ensuring consistent
behavior regardless of the frontend.

This module will be implemented after Phase 1 (router) and Phase 2 (generator)
are complete.
"""

# TODO: Phase 2-3 implementation
# - run_query(book_name, question) -> answer dict
# - Orchestrates: load index -> route to sections -> generate answer
# - Returns structured result with answer text and any relevant images
