"""Compatibility entrypoint; deploy interfaces/streamlit_app.py for new setups."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from streamlit_app import *  # noqa: F401,F403
