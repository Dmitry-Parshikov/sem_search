from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Literal

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from app.core.types import Chunk, RetrievedCandidate, Vector
from app.vector_store.base import VectorStore

_DISTANCE_MAP = {"cosine": Distance.COSINE}


class QdrantVectorStore(VectorStore):
    """`qdrant_client.QdrantClient` wrapper.

    Embedded mode (`path` set) and remote mode (`url` set) are mutually
    exclusive -- exactly one must be provided, matching
    `VectorStoreConfig.qdrant.mode`.

    Qdrant point IDs must be int or UUID, not arbitrary strings, so `chunk_id`
    is mapped to a deterministic UUID5 (derived from the chunk_id string) for
    the point id; the real `chunk_id`/`doc_id`/text/metadata are kept in the
    point payload so results can be reconstructed by `search()`.
    """

    def __init__(self, path: str | None = None, url: str | None = None) -> None:
        if bool(path) == bool(url):
            raise ValueError("QdrantVectorStore requires exactly one of `path` or `url`")
        if path is not None:
            resolved_path = Path(path).resolve()
            resolved_path.mkdir(parents=True, exist_ok=True)
            self._client = QdrantClient(path=str(resolved_path))
        else:
            self._client = QdrantClient(url=url)

    def create_collection(self, name: str, dimension: int, distance: Literal["cosine"] = "cosine") -> None:
        if self.collection_exists(name):
            return
        self._client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=dimension, distance=_DISTANCE_MAP[distance]),
        )

    def upsert(self, collection: str, chunks: list[Chunk], vectors: list[Vector]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")

        points = [
            PointStruct(
                id=_point_id(chunk.chunk_id),
                vector=vector,
                payload={
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "text": chunk.text,
                    "position": chunk.position,
                    "metadata": dict(chunk.metadata),
                },
            )
            for chunk, vector in zip(chunks, vectors)
        ]
        if points:
            self._client.upsert(collection_name=collection, points=points)

    def search(
        self,
        collection: str,
        query_vector: Vector,
        top_k: int,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[RetrievedCandidate]:
        query_filter = _build_filter(metadata_filter)
        response = self._client.query_points(
            collection_name=collection,
            query=query_vector,
            limit=top_k,
            query_filter=query_filter,
        )
        hits = response.points

        results: list[RetrievedCandidate] = []
        for rank, hit in enumerate(hits):
            payload = hit.payload or {}
            results.append(
                RetrievedCandidate(
                    chunk_id=payload.get("chunk_id", str(hit.id)),
                    doc_id=payload.get("doc_id", ""),
                    text=payload.get("text", ""),
                    score=float(hit.score),
                    metadata=payload.get("metadata", {}),
                    rank=rank,
                )
            )
        return results

    def delete_collection(self, name: str) -> None:
        if self.collection_exists(name):
            self._client.delete_collection(collection_name=name)

    def collection_exists(self, name: str) -> bool:
        try:
            return self._client.collection_exists(collection_name=name)
        except Exception:
            return False

    def health(self) -> bool:
        try:
            self._client.get_collections()
            return True
        except Exception:
            return False


def _point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def _build_filter(metadata_filter: dict[str, Any] | None) -> Filter | None:
    if not metadata_filter:
        return None
    conditions = [
        FieldCondition(key=f"metadata.{key}", match=MatchValue(value=value))
        for key, value in metadata_filter.items()
    ]
    return Filter(must=conditions)
