"""Reusable PDF/TXT loading and character-based document chunking."""

from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

if TYPE_CHECKING:
    from knowledge_base.storage import ChromaStore


class IngestionError(ValueError):
    """A source could not be read or parsed; safe to present to a caller."""


class UnsupportedFormatError(IngestionError):
    """The source is not a PDF or TXT file."""


def load_document(path: str | Path) -> list[Document]:
    """Load UTF-8 TXT or PDF text, omitting whitespace-only content.

    Metadata contains ``source`` (the filename) and, for PDFs, ``page``
    (the original one-based page number). Empty files and PDFs without
    extractable text return an empty list. OCR is not performed.
    Read/decode/parser failures raise IngestionError with the original cause.
    A PDF parsing failure rejects the whole file rather than returning partial text.
    """
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix not in {".pdf", ".txt"}:
        raise UnsupportedFormatError(
            f"Unsupported format for '{source.name}': expected PDF or TXT."
        )

    try:
        if source.stat().st_size == 0:
            return []
        if suffix == ".txt":
            text = source.read_text(encoding="utf-8-sig")
            return (
                [Document(page_content=text, metadata={"source": source.name})]
                if text.strip()
                else []
            )

        documents = []
        with source.open("rb") as stream:
            reader = PdfReader(stream)
            if reader.is_encrypted and not reader.decrypt(""):
                raise IngestionError(
                    f"Cannot read '{source.name}': PDF requires a password."
                )
            for page_number, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                if text.strip():
                    documents.append(
                        Document(
                            page_content=text,
                            metadata={"source": source.name, "page": page_number},
                        )
                    )
        return documents
    except IngestionError:
        raise
    except Exception as exc:
        # PDF parsers can raise several exception types on damaged input.
        # Normalize them at the loading boundary without discarding the cause.
        raise IngestionError(f"Could not parse '{source.name}': {exc}") from exc


def split_documents(
    documents: Iterable[Document],
    *,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[Document]:
    """Split loaded documents independently, preserving all source metadata.

    Sizes count Python characters. Overlap is a target: natural separator
    boundaries can produce less overlap. PDF chunks never cross page boundaries.
    """
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")
    if (
        isinstance(chunk_overlap, bool)
        or not isinstance(chunk_overlap, int)
        or not 0 <= chunk_overlap < chunk_size
    ):
        raise ValueError("chunk_overlap must be an integer from 0 to chunk_size - 1.")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    return splitter.split_documents(
        document for document in documents if document.page_content.strip()
    )


def ingest_document(
    path: str | Path,
    *,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[Document]:
    """Load a supported file and return metadata-preserving text chunks."""
    return split_documents(
        load_document(path), chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )


def index_document(
    path: str | Path,
    store: "ChromaStore",
    *,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[str]:
    """Parse, chunk, and persist a document; return newly indexed chunk IDs."""
    return store.index_chunks(
        ingest_document(path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    )


def replace_document(
    filename: str,
    path: str | Path,
    store: "ChromaStore",
    *,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    batch_size: int = 128,
) -> list[str]:
    """Reprocess a file and replace the exact indexed filename, without renaming it.

    The replacement path may have a different basename. Source metadata retains
    the selected document identity; PDF page metadata comes from the new file.
    Parsing failures leave storage unchanged; empty replacements are rejected.
    """
    chunks = ingest_document(path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    documents = [
        Document(page_content=chunk.page_content, metadata={**chunk.metadata, "source": filename})
        for chunk in chunks
    ]
    return store.replace_chunks(filename, documents, batch_size=batch_size)
