# Repository Guidelines

## Project Structure & Module Organization
This repository is a Python-based HUCE assistant with two main areas:
- `backend/`: data loading, ChromaDB seeding, retriever, tools, and agent logic.
- `frontend/`: Chainlit UI, chat events, and frontend-specific helpers.
- `backend/data/`: source documents used for retrieval. Treat these as curated inputs, not app code.
- `backend/app/database/cache/`: generated pickle cache files.
- Root files such as `main.py`, `requirements.txt`, and `README.md` are the main entrypoints and setup notes.

## Build, Test, and Development Commands
- `pip install -r requirements.txt`: install runtime dependencies.
- `python main.py`: seed ChromaDB and run the CLI-style agent check defined in the root entrypoint.
- `chainlit run frontend/main.py -w`: start the chat UI during development.
- `docker run -d --name chromadb_server -p 8080:8000 -v ./chroma_data:/chroma/chroma chromadb/chroma`: start the local ChromaDB server expected by the backend.
- `python -c "from backend.app.database.seed_data import setup; setup()"`: call the seeding routine directly when adjusting documents or embeddings.

## Coding Style & Naming Conventions
Use Python 3 style with 4-space indentation, `snake_case` for functions and modules, and `PascalCase` for classes and dataclasses. Keep functions small and explicit. Prefer clear Vietnamese or English identifiers that match existing code. There is no formatter configured in the repo, so follow the surrounding style and keep imports grouped logically.

## Testing Guidelines
There is no formal test suite checked in yet. When adding tests, prefer `pytest` and place files under `tests/` with names like `test_*.py`. For agent changes, validate manually by:
- running `python main.py`
- asking a few representative retrieval questions
- confirming the frontend still starts with Chainlit

## Commit & Pull Request Guidelines
Git history is currently minimal and uses short imperative commits such as `fix` or `third commit`. For new work, use concise messages like `fix retriever fallback` or `add seed data filter`. Pull requests should include:
- a short summary of what changed
- how it was tested
- any required config changes such as `.env` keys or ChromaDB setup
- screenshots or short clips for frontend changes

## Security & Configuration Tips
Keep secrets in `.env` only. Common required variables include `GEMINI_API_KEY`, `GROQ_API_KEY`, `HF_TOKEN`, and `CHROMA_HOST`/`CHROMA_PORT`. Do not commit generated caches, local virtual environments, or downloaded model artifacts.
