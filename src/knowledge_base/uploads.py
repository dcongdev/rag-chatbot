"""Adapt uploaded file bytes to the existing path-based ingestion pipeline."""

from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory

from knowledge_base.ingestion import UnsupportedFormatError, index_document, replace_document
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


def replace_upload(
    target_filename: str, filename: str, content: bytes, store: ChromaStore
) -> list[str]:
    """Replace the selected source using a temporary uploaded PDF/TXT file."""
    name = PurePosixPath(filename.replace("\\", "/")).name
    if Path(name).suffix.lower() not in {".pdf", ".txt"}:
        raise UnsupportedFormatError("Upload a PDF or TXT file.")
    with TemporaryDirectory() as directory:
        path = Path(directory) / name
        path.write_bytes(content)
        return replace_document(target_filename, path, store)
