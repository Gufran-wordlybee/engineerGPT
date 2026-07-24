# Engineering Book RAG Assistant — Project Plan

## Motivation

- Studying from 1000+ page engineering textbooks means exams are drawn directly from book content, so accurate, book-grounded answers matter more than general knowledge.
- Current manual workflow is slow and clunky: Cmd+F a term, read the surrounding paragraph, or split the screen, take a screenshot, and paste it into ChatGPT.
- Goal: replace that manual workflow with a tool where the book is pre-loaded, the student selects it, and just asks questions directly.

## Core Requirements

1. Support engineering books of ~1000+ pages.
2. Answers must be grounded in the book content (exams are set from the book, not general knowledge).
3. Books are pre-loaded — user just selects a book and starts querying, no upload step.
4. Books will be updated weekly according to the curriculum.
5. Must handle math-heavy equations and diagrams/images — approach still being worked out.
6. No citation of exact source snippets — deliberately skipped to avoid citing the wrong section and creating credibility issues (e.g., in interviews). Priority instead is **topic relevance**: if asked about Topic A, the answer must be about Topic A, not an adjacent Topic B.
7. Start as a CLI tool, then deploy via Streamlit (or similar) once core logic is proven.

## Known Problems / Constraints

- **Scanned PDFs**: Not all book PDFs are text-based; some are scanned images. This is handled outside the pipeline — books will be pre-converted to clean, text-based PDFs before being added to the system. The pipeline always receives clean text PDFs as input.
- **Equations & diagrams**: No OCR-to-LaTeX approach planned, since OCR tends to mangle math notation. Current thinking: treat equations and diagrams the same way — extract them as **images** during preprocessing and hand them to a vision-capable model at answer time, rather than trying to represent them as text.
- **No citations**: Deliberate decision. Citation would require pinpoint-accurate retrieval to avoid citing the wrong section, which isn't reliable enough at this stage. Priority is keeping answers on-topic, not sourcing exact quotes.

## Architecture Decision: Vector DB vs. Vectorless

**Decision: Vectorless (structure-based retrieval), not a vector database.**

Reasoning:
- Textbooks already have a reliable structure — Table of Contents, chapters, sections. This structure is more precise for topic-matching than embedding similarity, which often confuses semantically adjacent subtopics (a major risk given the "no wrong-topic answers" requirement).
- Vector chunking tends to break apart equations, multi-page derivations, and figure+caption pairs — exactly the content this project needs to preserve intact.
- At the project's actual scale (a fixed, curated set of curriculum books, updated weekly — not millions of arbitrary documents), a vector database adds infrastructure overhead without a clear accuracy benefit.
- Approach instead: split each book into sections by TOC, use an LLM-based "router" to pick the correct section(s) for a given question, then feed the full section content (text + linked images/equations) to the answering model.

This mirrors the manual Cmd+F workflow that motivated the project in the first place — just automated and scaled to full sections instead of single paragraphs.

---

## Implementation Plan (Phases)

### Phase 0 — Book Preprocessing Pipeline
Run once per book (not per query). This is the foundation everything else depends on.

1. Input is always a clean, text-based PDF (scanned/dirty PDFs are cleaned up manually before this step).
2. Extract the Table of Contents — from PDF bookmarks if available, or by detecting heading patterns (chapter titles, numbered sections like 3.2, 3.2.1) via regex/font-size heuristics.
3. Split the book into section-level chunks based on TOC boundaries. Each chunk = one section, stored with metadata (chapter, section number, title, page range).
4. For each section, separately extract: plain text, equation regions (as images), diagram/figure images (linked to their section).
5. Save everything as structured files (JSON per section) — no database needed at this stage.

**Exit criteria:** Any section of the book can be reliably retrieved as clean text + linked images, verified across at least 3 different books.

### Phase 1 — Section Router
Built as `core/router.py`; live accuracy evaluation is still required before Phase 2.

1. Builds an in-memory flattened index per book from the hierarchical `index.json`.
2. On a query, an LLM reads the compact section catalog (not the full book) and returns the best-matching section(s).
3. Supports returning multiple sections for cross-chapter/comparison questions.
4. Evaluation lives in `evaluation/eval_router.py` with hand-labeled question sets under `evaluation/questions/`.

**Exit criteria:** On 15-20 sample questions per book, router selects the correct section 90%+ of the time. This is the single most important accuracy checkpoint in the project — test manually before proceeding.

### Phase 2 — Answer Generation

1. Take the router's selected section(s) and build a prompt: system instructions (exam-focused study assistant, no fabrication) + section text + equations + user's question.
2. If the section includes diagrams/equations relevant to the query, send the associated images to a vision-capable model alongside the text.
3. No citation step, per requirement 6 above — answer directly from section content.

**Exit criteria:** Given a correctly-routed section, answers are accurate and don't introduce information beyond what's in the section.

### Phase 3 — CLI Interface

1. Flow: list available books → user selects one → user asks a question → pipeline (router → generator) runs → answer is printed.
2. First point where Phase 0/1/2 are wired together end-to-end.

**Exit criteria:** Full flow works across at least 2 books, 10+ queries each, without manual intervention mid-run.

### Phase 4 — Weekly Book Update Flow

1. Adding a new book should mean: drop the clean PDF into a folder, run the Phase 0 preprocessing script — nothing else.
2. Confirm this is purely additive — adding a new book must not require reprocessing existing books.

**Exit criteria:** A newly added book is queryable after one script run, with no manual fixes.

### Phase 5 — Streamlit Deployment

1. Same underlying pipeline as the CLI, wrapped in a Streamlit UI: book selector, question input, answer display, image display area for diagrams/equations.
2. Mostly UI work — core logic is already validated via CLI by this point.

**Exit criteria:** Demoable live via browser link, no CLI knowledge required from the viewer.

---

## Folder Structure

```
engineering-rag-assistant/
│
├── books/
│   ├── raw/                    # pre-cleaned text PDFs go here (input)
│   └── processed/              # Phase 0 output, one subfolder per book
│       └── <book_name>/
│           ├── sections/       # one JSON per section: {text, equations, images, metadata}
│           ├── images/         # extracted diagram/equation image files
│           └── index.json      # flattened TOC used by the router in Phase 1
│
├── preprocessing/
│   ├── extract_toc.py          # Phase 0: detect chapter/section boundaries
│   ├── split_sections.py       # Phase 0: split book into section-level chunks
│   ├── extract_images.py       # Phase 0: pull out diagrams/equation images per section
│   └── build_index.py          # Phase 0: generate index.json from processed sections
│
├── core/
│   ├── router.py                # Phase 1: LLM call that picks section(s) from index.json
│   ├── generator.py             # Phase 2: builds prompt, calls LLM (+ vision) for final answer
│   └── pipeline.py              # ties router.py + generator.py together as one callable flow
│
├── interfaces/
│   ├── cli.py                   # Phase 3: command-line entrypoint
│   └── app_streamlit.py         # Phase 5: Streamlit entrypoint
│
├── config/
│   └── settings.py               # model names, paths, constants — centralized
│
├── .env                          # API keys (never committed)
├── .gitignore                    # excludes .env, books/processed/, __pycache__
├── requirements.txt
└── README.md
```

**Why this structure:**
- `books/raw` vs `books/processed` separates input from pipeline output, so re-running preprocessing never risks touching source PDFs.
- `preprocessing/` is separate from `core/` because preprocessing runs once per book (weekly), while `core/` runs per query (constantly). This avoids ever reprocessing a whole book just to answer one question.
- `core/pipeline.py` exists so both `cli.py` and `app_streamlit.py` call the same underlying logic — written once, reused by both interfaces.
- `config/settings.py` centralizes model names and paths so switching models (e.g., free tier to paid) is a one-line change, not a find-and-replace across files.
- `.env` keeps secrets out of code; `settings.py` just reads from it.

## Environment Variables

```
GROQ_LLM_API_KEY=          # Groq API key for index abstracts/router
GROQ_LLM_MODEL_NAME=       # Groq model string, kept configurable
ROUTER_LLM_MODEL_NAME=     # optional router-specific override
ROUTER_TOP_K=3             # optional router result count
VISION_MODEL_NAME=         # may be same as above if provider handles both text+vision
BOOKS_RAW_PATH=./books/raw
BOOKS_PROCESSED_PATH=./books/processed
```

No database URL needed — everything is stored as JSON/files on disk, consistent with the vectorless approach.

## Time Estimate (solo, with LLM assistance for coding)

| Phase | Estimate | Notes |
|---|---|---|
| Phase 0 — Preprocessing | 4-6 days | Hardest part; TOC detection varies across book formats |
| Phase 1 — Router | Built, eval pending | Needs live 90%+ accuracy run |
| Phase 2 — Generation | 3-4 days | Vision model integration + prompt tuning |
| Phase 3 — CLI | 1 day | Should be fast if Phase 0-2 work |
| Phase 4 — Weekly updates | 1 day | Confirming reusability of Phase 0 |
| Phase 5 — Streamlit | 2-3 days | UI work, image display handling |
| **Total** | **~2.5-3 weeks** | Add buffer for varied book formatting |

**Recommendation:** Don't try to perfect Phase 0 for every possible book format upfront. Get it working for 2-3 books, build Phases 1-3 against those, and only harden preprocessing for edge cases after the full pipeline is proven end-to-end.
