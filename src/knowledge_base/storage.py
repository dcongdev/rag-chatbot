"""Persistent chunk indexing and vector queries through a shared local client."""

import hashlib
import json
import math
import os
from collections.abc import Iterable
from pathlib import Path

import chromadb
from chromadb.config import Settings
from chromadb.api.types import QueryResult
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from knowledge_base.embeddings import MODEL_NAME, create_embeddings


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
