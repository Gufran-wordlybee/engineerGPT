# Adding books and deploying EngineerGPT

The website has two sources of data:

```text
GitHub repository                         Supabase database
books/processed/<book_id>/  ---------->   books table
section text, index, images               which books appear in the sidebar
                                           chat threads and messages
```

The folder contains the textbook itself. The database only says that the folder is ready to show and saves chats. A book appears only when both exist: its processed folder is pushed to GitHub and it has a row in Supabase.

## One-time setup

1. Create a free [Supabase](https://supabase.com) project.
2. Install the local dependencies from `Phase-0/` (this includes the optional heavy preprocessing stack):

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. In Supabase, open **SQL Editor**, create a query, paste and run [`../interfaces/schema.sql`](../interfaces/schema.sql). This creates the book/chat tables and turns on Row Level Security.
4. Copy [`../.env.example`](../.env.example) to `Phase-0/.env`, then fill in the values below. Do not commit this file.
5. For Streamlit Community Cloud, put the same values in **App settings → Secrets**:

```toml
GROQ_LLM_API_KEY = "gsk_..."
GROQ_LLM_MODEL_NAME = "llama-3.3-70b-versatile"
GENERATE_LLM_MODEL = "qwen/qwen3.6-27b"
SUPABASE_URL = "https://your-project.supabase.co"
SUPABASE_SERVICE_ROLE_KEY = "your-service-role-key"
```

The Streamlit app copies those secrets into environment variables before the existing backend loads. The service-role key is safe in Streamlit Secrets because it stays on the server; never put it in browser code or GitHub.

## API keys you need

- **Groq API key** (`GROQ_LLM_API_KEY`): required for routing and generating answers. Create it at [Groq Console](https://console.groq.com/keys).
- **Supabase project URL** (`SUPABASE_URL`) and **service-role key** (`SUPABASE_SERVICE_ROLE_KEY`): required for the sidebar book registry and persistent chats. Find them in Supabase **Project Settings → API**.

No OpenAI key is needed: the installed OpenAI Python library talks to Groq's compatible API. Gemini is not needed for the deployed frontend either; it is only relevant if you enable optional local preprocessing features.

## Adding a book

Run these commands from `Phase-0/`.

1. Put the PDF at `books/raw/<book_id>.pdf`. Use a simple folder-safe ID such as `digital_logic`.
2. Process it locally:

   ```bash
   python -m preprocessing.run_pipeline books/raw/<book_id>.pdf
   ```

3. Verify that `books/processed/<book_id>/index.json` exists and that `total_sections` is greater than zero.
4. Register the book in Supabase. The ID must exactly match the processed folder name:

   ```bash
   python -m interfaces.register_book <book_id> "Human readable title"
   ```

5. Commit the processed book and push it. It is deliberately not ignored by Git:

   ```bash
   cd ..
   git add Phase-0/books/processed/<book_id>
   git commit -m "Add <book_id> book"
   git push
   ```

6. Streamlit Community Cloud redeploys after the push. If it does not, use **Reboot app** in its dashboard.

## Deploying the app

1. Push this project to GitHub. A public repository works with the Community Cloud free tier; never commit `.env` or secrets.
2. Create a Streamlit Community Cloud app. Set the main file to `Phase-0/interfaces/streamlit_app.py`.
3. Its repository-root `requirements.txt` is intentionally slim: it excludes `marker-pdf` and Torch because preprocessing never runs in the deployed app.
4. Paste the secrets shown above, save, and reboot the app.

## Troubleshooting

- **Book does not show:** check that `register_book.py` succeeded, then confirm the processed folder was committed and the cloud app redeployed.
- **“Content missing” warning:** Supabase knows the book but GitHub does not yet contain `books/processed/<book_id>/index.json` in the deployed commit.
- **No diagrams/equations:** ensure `books/processed/<book_id>/images/` was committed too. The current `ai` and `coa` folders contain images locally; verify they are included in the GitHub commit as well.
- **Cannot generate an answer:** check the Groq key and model names in Streamlit Secrets, then reboot the app.

## Repository size

Keep each image far below GitHub's 100 MB per-file limit. A few books are fine in Git, but if processed books grow beyond a few hundred MB, move `images/*.png` to Git LFS or object storage (such as Supabase Storage) and keep their URLs in the section JSON. That scaling path is intentionally not implemented yet.
