# `config/` — Centralized Settings

All tunable constants and environment-variable-driven configuration live here. Every other module imports from `config.settings` — no magic numbers scattered across the codebase.

## How It Works

1. `settings.py` loads `.env` from the project root on import
2. Environment variables override defaults (useful for CI, Docker, per-user config)
3. All settings are typed Python constants — IDE autocomplete works

## Settings Reference

### LLM / API Settings

| Setting | Env Var | Default | Description |
|---------|---------|---------|-------------|
| `LLM_API_KEY` | `LLM_API_KEY` | `""` | API key for LLM provider (OpenAI, Anthropic, etc.) |
| `LLM_MODEL_NAME` | `LLM_MODEL_NAME` | `""` | Model name (e.g. `gpt-4o-mini`, `claude-sonnet-4-20250514`) |
| `VISION_MODEL_NAME` | `VISION_MODEL_NAME` | `""` | Vision model for Phase 2 diagram understanding |
| `LLM_ABSTRACTS_ENABLED` | `LLM_ABSTRACTS_ENABLED` | `false` | Enable LLM-generated section abstracts at index build time |

### Router Settings (Phase 1)

| Setting | Env Var | Default | Description |
|---------|---------|---------|-------------|
| `ROUTER_MODEL_NAME` | `ROUTER_MODEL_NAME` | Same as `LLM_MODEL_NAME` | Override model for routing (can use cheaper model) |
| `ROUTER_TEMPERATURE` | `ROUTER_TEMPERATURE` | `0` | Sampling temperature (0 = deterministic) |
| `TWO_STAGE_THRESHOLD` | `TWO_STAGE_THRESHOLD` | `30` | Section count above which two-stage routing is used |
| `ROUTER_TOP_K_DEFAULT` | `ROUTER_TOP_K_DEFAULT` | `2` | Default number of sections returned by router |

### Paths

| Setting | Env Var | Default | Description |
|---------|---------|---------|-------------|
| `BOOKS_RAW_PATH` | `BOOKS_RAW_PATH` | `books/raw/` | Where to look for input PDFs |
| `BOOKS_PROCESSED_PATH` | `BOOKS_PROCESSED_PATH` | `books/processed/` | Where preprocessed output goes |

### Preprocessing Constants

| Setting | Default | Used By | Description |
|---------|---------|---------|-------------|
| `MIN_HEADING_FONT_RATIO` | `1.2` | `extract_toc` | Minimum font-size ratio for heading detection |
| `MIN_SECTION_CHARS` | `200` | `split_sections` | Sections below this merge into siblings |
| `TOP_N_KEYWORDS` | `15` | `build_index` | TF-IDF keywords per section |
| `MIN_IMAGE_SIZE` | `50` | `extract_images` | Minimum image dimension in pixels |

See [`settings.py`](settings.py) for the complete list with detailed comments.

## `.env` Example

Create `Phase-0/.env`:
```env
# Required for LLM features
LLM_API_KEY=sk-your-key-here
LLM_MODEL_NAME=gpt-4o-mini

# Optional overrides
ROUTER_MODEL_NAME=gpt-4o-mini
ROUTER_TEMPERATURE=0
TWO_STAGE_THRESHOLD=30
LLM_ABSTRACTS_ENABLED=true
```
