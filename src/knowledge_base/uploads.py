"""Adapt uploaded file bytes to the existing path-based ingestion pipeline."""

from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory

from knowledge_base.ingestion import UnsupportedFormatError, index_document
from knowledge_base.storage import ChromaStore


def index_upload(filename: str, content: bytes, store: ChromaStore) -> list[str]:
    """Index a temporary upload, retaining only its basename as source metadata."""
    name = PurePosixPath(filename.replace("\\", "/")).name
    if Path(name).suffix.lower() not in {".pdf", ".txt"}:
        raise UnsupportedFormatError("Upload a PDF or TXT file.")
    with TemporaryDirectory() as directory:
        path = Path(directory) / name
        path.write_bytes(content)
        return index_document(path, store)
