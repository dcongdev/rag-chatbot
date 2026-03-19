"""Semantic retrieval of indexed chunks, without answer generation."""

from dataclasses import dataclass
from typing import Any

from knowledge_base.storage import ChromaStore


@dataclass(frozen=True)
class RetrievedChunk:
    """A matching chunk with its provenance and optional distance-based score.

    Distance is Chroma's raw metric (squared L2 for the existing collection).
    Smaller distances are better. similarity_score = 1 / (1 + distance), so
    larger scores are better; this is neither cosine similarity nor confidence.
    Negative floating-point distance artifacts are clamped to zero for scoring.
    """

    id: str
    content: str
    metadata: dict[str, Any]
    distance: float | None
    similarity_score: float | None


def retrieve(
    query: str, store: ChromaStore, *, top_k: int = 4
) -> list[RetrievedChunk]:
    """Return up to top_k chunks in nearest-first order from an existing store.

    Blank queries and empty collections return []. Invalid query types or top_k
    values raise ValueError. Storage/model failures propagate to the caller.
    No model, client, or collection is created independently of the supplied store.
    """
    result = store.query_chunks(query, top_k=top_k)
    if result is None:
        return []
    ids = result["ids"][0]
    contents = (result.get("documents") or [[]])[0]
    metadatas = (result.get("metadatas") or [[None] * len(ids)])[0]
    distances = (result.get("distances") or [[None] * len(ids)])[0]
    chunks = []
    for identifier, content, metadata, distance in zip(
        ids, contents, metadatas, distances, strict=True
    ):
        if content is None:
            continue
        chunks.append(
            RetrievedChunk(
                id=identifier,
                content=content,
                metadata=dict(metadata or {}),
                distance=float(distance) if distance is not None else None,
                similarity_score=(
                    1.0 / (1.0 + max(0.0, distance)) if distance is not None else None
                ),
            )
        )
    return chunks
