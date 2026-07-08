# `interfaces/` — User-Facing Frontends

Both frontends call the same `core.pipeline` module, ensuring consistent behavior regardless of how the user interacts with the system.

## Module Overview

| Module | Phase | Status | Description |
|--------|-------|--------|-------------|
| [`cli.py`](cli.py) | 3 | 🔲 Planned | Interactive terminal Q&A loop |
| [`app_streamlit.py`](app_streamlit.py) | 5 | 🔲 Planned | Web UI with book selector + image display |

## Planned Features

### CLI (Phase 3)
- List available preprocessed books
- Select a book → enter interactive question loop
- Display answers with section references
- Entry point: `python -m interfaces.cli`

### Streamlit (Phase 5)
- Book selector dropdown (populated from `books/processed/`)
- Question text input
- Rendered markdown answers
- Image carousel for diagrams/equations from matched sections
- Entry point: `streamlit run interfaces/app_streamlit.py`

## Architecture

```
              ┌──────────┐      ┌───────────────┐
  Terminal ──►│  cli.py   │─────►│               │
              └──────────┘      │ core.pipeline  │──► answer
              ┌──────────┐      │               │
  Browser ──►│ streamlit │─────►│               │
              └──────────┘      └───────────────┘
```

Both interfaces are thin wrappers — all logic lives in `core/`.
