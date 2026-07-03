# Engineering RAG Assistant

A vectorless, structure-based RAG system for studying engineering textbooks. Uses TOC-aware section routing instead of vector search — because textbook structure (chapters, sections, headings) is more reliable for topic-matching than embedding similarity.

## Why Vectorless?

- Textbooks already have reliable structure (TOC, chapters, sections)
- Vector chunking breaks equations, derivations, and figure+caption pairs
- At curriculum scale (~10 books), vector DB adds overhead without accuracy benefit
- Structure-based routing mirrors the manual "Cmd+F → read the section" workflow

## Setup

```bash
# Clone and enter
git clone <repo-url>
cd engineering-rag-assistant

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env .env.local  # Edit with your API keys
```

## Usage

### Phase 0: Preprocess a Book

Drop a clean text-based PDF into `books/raw/`, then run:

```bash
# Process a single book
python -m preprocessing.run_pipeline books/raw/your_book.pdf

# Process all books in books/raw/
python -m preprocessing.run_pipeline

# Force reprocessing of an already-processed book
python -m preprocessing.run_pipeline books/raw/your_book.pdf --force
```

### Standalone Module Testing

```bash
# Test TOC extraction only
python -m preprocessing.extract_toc books/raw/your_book.pdf
```

## Folder Structure

```
books/raw/           → Drop clean text-based PDFs here (input)
books/processed/     → Pipeline output: sections, images, index per book
preprocessing/       → Runs once per book (weekly): TOC, split, images, index
core/                → Runs per query: router, generator, pipeline
interfaces/          → CLI and Streamlit entrypoints
config/              → Centralized settings and constants
```

## Phase Roadmap

| Phase | Status | Description |
|-------|--------|-------------|
| **0 — Preprocessing** | ✅ Built | TOC extraction, section splitting, image extraction, index building |
| **1 — Router** | 🔲 Planned | LLM-based section selection from index.json |
| **2 — Generator** | 🔲 Planned | Answer generation with vision model support |
| **3 — CLI** | 🔲 Planned | Interactive command-line question/answer loop |
| **4 — Weekly Updates** | 🔲 Planned | Drop-in book addition workflow |
| **5 — Streamlit** | 🔲 Planned | Web UI for browser-based access |

## Requirements

- Python 3.10+
- PyMuPDF (for PDF processing)
- python-dotenv (for environment management)
