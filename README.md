# AI Knowledge Base Assistant

A lightweight Python scaffold for an AI Knowledge Base Assistant. Application
logic and UI are intentionally left as placeholders.

## Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run
from the repository root:

```powershell
uv python install 3.11
uv sync --locked
Copy-Item .env.example .env
```

Python 3.11 is pinned for development; the project supports Python 3.11+.
`uv sync` creates `.venv` and installs the project and its locked dependencies.
Replace the placeholder `GEMINI_API_KEY` in `.env` when needed. Environment-file
loading and Gemini integration are not implemented yet.

## Run

```powershell
uv run streamlit run app.py
```

The entry point is a placeholder, so Streamlit currently displays an empty page.

## Structure

```text
app.py                     # Placeholder Streamlit entry point
src/knowledge_base/
    __init__.py
    ingestion.py           # Placeholder
    retrieval.py           # Placeholder
    generation.py          # Placeholder
tests/                     # Placeholder for future tests
```

## Verify the environment

```powershell
uv run python --version
uv run python -c "import streamlit; import knowledge_base; print('Environment ready')"
```
