# AI Knowledge Base Assistant

A lightweight modular AI Knowledge Base Assistant with PDF/TXT ingestion and
text chunking. Retrieval, generation, and UI remain placeholders.

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
    ingestion.py           # PDF/TXT parsing and text chunking
    retrieval.py           # Placeholder
    generation.py          # Placeholder
tests/                     # Automated ingestion tests
```

## Verify the environment

```powershell
uv run python --version
uv run python -c "import streamlit; import knowledge_base; print('Environment ready')"
```

## Document ingestion

```python
from knowledge_base.ingestion import IngestionError, ingest_document

try:
    chunks = ingest_document("notes.pdf", chunk_size=1000, chunk_overlap=200)
except IngestionError as error:
    print(error)
else:
    for chunk in chunks:
        print(chunk.metadata, chunk.page_content)
```

`load_document(path)` and `split_documents(documents, ...)` can also be used
independently. They return LangChain `Document` objects for future retrieval use.
PDF extraction uses [pypdf](https://pypdf.readthedocs.io/en/stable/);
chunking uses LangChain's
[RecursiveCharacterTextSplitter](https://reference.langchain.com/python/langchain-text-splitters/character/RecursiveCharacterTextSplitter).

- Supported extensions are `.pdf` and `.txt` (case insensitive). TXT files must
  use UTF-8; an optional UTF-8 BOM is removed.
- Metadata includes `source` (filename), and PDF `page` (one-based original page
  number). Each PDF page is split independently; chunks never span pages.
- Chunk sizes count characters, with overlap determined by natural separator
  boundaries, up to the configured target. Size must be positive and overlap
  must be nonnegative and smaller than size.
- Empty files, blank pages, and PDFs without extractable text yield no chunks.
  Scanned PDFs require OCR, which is not implemented.
- Unsupported formats raise `UnsupportedFormatError`, a subclass of
  `IngestionError`. Read, decode, password, and parsing failures raise
  `IngestionError`; a failed PDF is rejected as a whole.

Run the focused tests (no network or API key required):

```powershell
uv run python -m unittest discover -s tests -v
```
