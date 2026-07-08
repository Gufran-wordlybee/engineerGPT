# `core/` — Query-Time Modules

This package runs **per question** (not per book). It takes a student's question plus a book name, routes it to the right section(s), and (in Phase 2) generates an answer.

## Module Overview

| Module | Phase | Status | Purpose |
|--------|-------|--------|---------|
| [`llm_client.py`](llm_client.py) | 1 | ✅ Built | Shared LLM caller (provider auto-detect, retry, backoff) |
| [`router.py`](router.py) | 1 | ✅ Built | Routes questions → best-matching section(s) via LLM |
| [`eval_router.py`](eval_router.py) | 1 | ✅ Built | Evaluation harness: measures routing accuracy |
| [`generator.py`](generator.py) | 2 | 🔲 Planned | Answer generation with vision model support |
| [`pipeline.py`](pipeline.py) | 3 | 🔲 Planned | Orchestrates router + generator into a single flow |

## Architecture

```
              ┌──────────────┐
  question ──►│   router.py  │──► section_ids + confidence
              │              │
              │  Stage 1:    │   (for large books)
              │  chapter pick│
              │      │       │
              │  Stage 2:    │
              │  section pick│
              │      │       │
              │  Fallback:   │
              │  TF-IDF kw   │
              └──────┬───────┘
                     │
              ┌──────▼───────┐
              │ llm_client.py│   shared by all modules
              │              │
              │ • OpenAI SDK │
              │ • Anthropic  │
              │ • Retry logic│
              └──────────────┘
```

## Quick Start

### Route a question (CLI)
```bash
cd Phase-0
python -m core.router fluid_mechanics "What is Bernoulli's equation?"
```

### Evaluate router accuracy
```bash
# Single book (needs books/processed/<book>/test_questions.json)
python -m core.eval_router fluid_mechanics

# All books with test questions
python -m core.eval_router
```

## Key Design Decisions

### Two-Stage Routing
Books with > `TWO_STAGE_THRESHOLD` sections (default: 30) use a two-stage approach:
1. **Stage 1**: Send only chapter titles + abstracts → LLM picks 1–2 chapters
2. **Stage 2**: Send the full subtree of those chapters → LLM picks final section(s)

This keeps each prompt small regardless of book size.

### Confusable-Pair Disambiguation
When candidate sections appear in `confusable_pairs.json`, the prompt explicitly warns the LLM about commonly confused sibling sections and their shared keywords. This targets the #1 requirement: no wrong-topic answers.

### Fallback Chain
1. LLM routing (two-stage or single-stage)
2. LLM retry with stricter prompt (on JSON parse failure)
3. TF-IDF keyword overlap (no LLM needed — cheap insurance)

### `llm_client.py`
Extracted from `build_index.py` to avoid duplication. Features:
- Auto-detects provider from model name (`claude` → Anthropic, else → OpenAI)
- Exponential backoff retry (1s, 2s, 4s)
- Returns `None` on permanent failure (auth errors, missing packages)
- Configurable `temperature`, `max_tokens`, `model`, `api_key`

## Test Questions Format

Create `books/processed/<book>/test_questions.json`:
```json
[
    {
        "question": "What is Bernoulli's equation and when is it applicable?",
        "expected_section": "3.2",
        "tags": ["confusable"]
    },
    {
        "question": "Explain the difference between laminar and turbulent flow",
        "expected_section": "5.1",
        "tags": []
    }
]
```

- Include 15–20 questions per book
- Tag ~5 questions as `"confusable"` (from confusable-pair sections)
- Target: ≥90% top-1 accuracy
