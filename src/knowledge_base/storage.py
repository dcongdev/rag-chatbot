"""Persistent chunk indexing and vector queries through a shared local client."""

import hashlib
import json
import math
import os
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import chromadb
from chromadb.config import Settings
from chromadb.api.types import QueryResult
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from knowledge_base.embeddings import MODEL_NAME, create_embeddings


@dataclass(frozen=True)
class IndexedDocument:
    filename: str
    chunk_count: int


class DocumentNotFoundError(LookupError):
    """No indexed chunks exist for the exact source filename."""


class ReplacementError(RuntimeError):
    """Replacement failed; the message indicates whether rollback succeeded."""


def _validate_filename(filename: str) -> None:
    if (
        not isinstance(filename, str) or not filename.strip()
        or filename in {".", ".."} or "/" in filename or "\\" in filename
    ):
        raise ValueError("Use an exact, nonempty source filename without directory paths.")


def chunk_id(document: Document) -> str:
    """Hash exact text and metadata, retaining distinct source/page provenance."""
    payload = json.dumps(
        {"text": document.page_content, "metadata": document.metadata},
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validate_metadata(document: Document) -> None:
    metadata = document.metadata
    if not isinstance(metadata.get("source"), str) or not metadata["source"].strip():
        raise ValueError("Each chunk must have a nonempty source filename.")
    if "page" in metadata and (
        type(metadata["page"]) is not int or metadata["page"] < 1
    ):
        raise ValueError("PDF page metadata must be a positive integer.")
    for key, value in metadata.items():
        if not isinstance(key, str) or not key:
            raise ValueError("Metadata keys must be nonempty strings.")
        if not isinstance(value, (str, int, float, bool)) or (
            isinstance(value, float) and not math.isfinite(value)
        ):
            raise ValueError("Storage metadata must contain finite scalar values.")


class ChromaStore:
    """Index chunks in a local collection using the fixed MiniLM embedding model.

    An injected LangChain Embeddings instance is intended for tests or an
    equivalent MiniLM adapter. Reuse the same model for a given collection.
    Use as a context manager to release local database resources.
    """

    def __init__(
        self,
        persist_directory: str | Path | None = None,
        *,
        collection_name: str = "knowledge_base",
        embeddings: Embeddings | None = None,
    ) -> None:
        self.persist_directory = Path(
            persist_directory
            if persist_directory is not None
            else os.environ.get("CHROMA_PERSIST_DIR", "data/chroma")
        ).expanduser().resolve()
        self._embeddings = embeddings
        self._closed = False
        self._client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(anonymized_telemetry=False),
        )
        try:
            self._collection = self._client.get_or_create_collection(
                name=collection_name,
                embedding_function=None,
                metadata={"embedding_model": MODEL_NAME},
            )
            if (self._collection.metadata or {}).get("embedding_model") != MODEL_NAME:
                raise ValueError("Collection uses a different or unknown embedding model.")
        except Exception:
            self._client.close()
            raise

    def index_chunks(
        self, chunks: Iterable[Document], *, batch_size: int = 128
    ) -> list[str]:
        """Return newly indexed IDs, skipping duplicates before embedding.

        Empty content is skipped. Metadata must contain a source filename and
        Chroma-compatible scalar values. Each batch is persisted automatically;
        if a later batch fails, retrying safely skips previously stored batches.
        """
        if self._closed:
            raise RuntimeError("The storage client is closed.")
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")
        unique = {}
        for document in chunks:
            if document.page_content.strip():
                _validate_metadata(document)
                unique.setdefault(chunk_id(document), document)
        ids = list(unique)
        added = []
        batch_size = min(batch_size, self._client.get_max_batch_size())
        for start in range(0, len(ids), batch_size):
            batch = ids[start : start + batch_size]
            existing = set(self._collection.get(ids=batch, include=[])["ids"])
            pending = [identifier for identifier in batch if identifier not in existing]
            if not pending:
                continue
            documents = [unique[identifier] for identifier in pending]
            vectors = self._get_embeddings().embed_documents(
                [document.page_content for document in documents]
            )
            self._collection.upsert(
                ids=pending,
                embeddings=vectors,
                documents=[document.page_content for document in documents],
                metadatas=[document.metadata for document in documents],
            )
            added.extend(pending)
        return added

    def _get_embeddings(self) -> Embeddings:
        if self._embeddings is None:
            self._embeddings = create_embeddings()
        return self._embeddings

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("The storage client is closed.")

    def list_documents(self) -> list[IndexedDocument]:
        """List exact, case-sensitive source filenames and unique chunk counts."""
        self._ensure_open()
        counts = Counter()
        offset = 0
        while True:
            records = self._collection.get(include=["metadatas"], limit=1000, offset=offset)
            for metadata in records["metadatas"]:
                if metadata and isinstance(metadata.get("source"), str):
                    counts[metadata["source"]] += 1
            if len(records["ids"]) < 1000:
                break
            offset += len(records["ids"])
        return [IndexedDocument(name, counts[name]) for name in sorted(counts)]

    def delete_document(self, filename: str) -> int:
        """Delete all chunks for one exact filename; missing documents raise."""
        self._ensure_open()
        _validate_filename(filename)
        where = {"source": {"$eq": filename}}
        ids = self._collection.get(where=where, include=[])["ids"]
        if not ids:
            raise DocumentNotFoundError(f"No indexed document named '{filename}'.")
        self._collection.delete(where=where)
        return len(ids)

    def replace_chunks(
        self, filename: str, chunks: Iterable[Document], *, batch_size: int = 128
    ) -> list[str]:
        """Replace an existing filename's chunks, returning all final chunk IDs.

        Validation precedes writes; originals remain during indexing. On failure,
        remove staged chunks and restore originals using saved embeddings. This
        compensates ordinary API failures, not process crashes or concurrent
        writers. An unavailable database may also prevent rollback; that failure
        is reported explicitly. Empty replacements are rejected.
        """
        self._ensure_open()
        _validate_filename(filename)
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")
        original = self._collection.get(
            where={"source": {"$eq": filename}},
            include=["documents", "metadatas", "embeddings"],
        )
        if not original["ids"]:
            raise DocumentNotFoundError(f"No indexed document named '{filename}'.")
        unique = {}
        for document in chunks:
            _validate_metadata(document)
            if document.metadata["source"] != filename:
                raise ValueError("Replacement chunks must match the target source filename.")
            if document.page_content.strip():
                unique.setdefault(chunk_id(document), document)
        if not unique:
            raise ValueError("Replacement must contain extractable, nonempty text.")
        old_ids = set(original["ids"])
        new_ids = list(unique)
        staged_ids = [identifier for identifier in new_ids if identifier not in old_ids]
        stale_ids = [identifier for identifier in original["ids"] if identifier not in unique]
        batch_size = min(batch_size, self._client.get_max_batch_size())
        deleting = False
        try:
            self.index_chunks(unique.values(), batch_size=batch_size)
            deleting = True
            for start in range(0, len(stale_ids), batch_size):
                self._collection.delete(ids=stale_ids[start : start + batch_size])
        except Exception as error:
            try:
                if deleting:
                    vectors = original["embeddings"].tolist()
                    for start in range(0, len(original["ids"]), batch_size):
                        end = start + batch_size
                        self._collection.upsert(
                            ids=original["ids"][start:end],
                            documents=original["documents"][start:end],
                            metadatas=original["metadatas"][start:end],
                            embeddings=vectors[start:end],
                        )
                for start in range(0, len(staged_ids), batch_size):
                    self._collection.delete(ids=staged_ids[start : start + batch_size])
            except Exception as rollback_error:
                raise ReplacementError(
                    "Replacement failed and rollback was incomplete. Inspect the document before retrying."
                ) from rollback_error
            raise ReplacementError("Replacement failed; the original document was preserved.") from error
        return new_ids

    def query_chunks(self, query: str, *, top_k: int) -> QueryResult | None:
        """Query stored vectors; return None for blank input or an empty collection.

        The same lazy embeddings instance serves indexing and query embedding.
        Invalid parameters raise ValueError; a closed client raises RuntimeError.
        """
        if self._closed:
            raise RuntimeError("The storage client is closed.")
        if not isinstance(query, str):
            raise ValueError("query must be a string.")
        if type(top_k) is not int or top_k <= 0:
            raise ValueError("top_k must be a positive integer.")
        query = query.strip()
        if not query:
            return None
        count = self._collection.count()
        if count == 0:
            return None
        vector = self._get_embeddings().embed_query(query)
        return self._collection.query(
            query_embeddings=[vector],
            n_results=min(top_k, count),
            include=["documents", "metadatas", "distances"],
        )

    def close(self) -> None:
        """Release the persistent client without deleting indexed data."""
        if not self._closed:
            self._client.close()
            self._closed = True

    def __enter__(self) -> "ChromaStore":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
