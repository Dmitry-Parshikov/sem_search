"""Offline indexing orchestration (Ф1.1-Ф1.6).

`IndexingService.run()` is always a full rebuild into a brand-new
`index_version` (never an incremental patch), per the reproducibility NFR
and plan decision #5 (each version gets its own Qdrant collection + BM25
pickle, so rollback is just a manifest pointer swap).
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from app.admin.manifest import IndexManifest
from app.admin.versioning import collection_name_for_version, compute_index_version, lexical_path_for_version
from app.chunking.base import Chunker
from app.config import Settings
from app.core.errors import IndexingError
from app.core.types import Chunk, Document
from app.embedding.base import Embedder
from app.lexical.base import LexicalIndex
from app.preprocessing.base import Preprocessor
from app.vector_store.base import VectorStore

LexicalIndexFactory = Callable[[], LexicalIndex]


class IndexingService:
    def __init__(
        self,
        embedder: Embedder,
        chunker: Chunker,
        vector_store: VectorStore,
        preprocessor: Preprocessor,
        lexical_index_factory: LexicalIndexFactory,
        settings: Settings,
    ) -> None:
        self._embedder = embedder
        self._chunker = chunker
        self._vector_store = vector_store
        self._preprocessor = preprocessor
        self._lexical_index_factory = lexical_index_factory
        self._settings = settings

    def run(self, documents: list[Document], source_corpus: str = "unknown") -> str:
        """Runs the full offline indexing pipeline and returns the new
        `index_version`."""

        try:
            return self._run(documents, source_corpus)
        except IndexingError:
            raise
        except Exception as exc:
            raise IndexingError(f"Indexing run failed: {exc}") from exc

    def _run(self, documents: list[Document], source_corpus: str) -> str:
        # Determinism: fixed processing order before chunking/batching.
        sorted_documents = sorted(documents, key=lambda d: d.doc_id)

        preprocessed_documents = [
            Document(
                doc_id=doc.doc_id,
                text=self._preprocessor.clean(doc.text, _content_type_of(doc)),
                source=doc.source,
                section=doc.section,
                date=doc.date,
                extra=doc.extra,
            )
            for doc in sorted_documents
        ]

        chunks: list[Chunk] = []
        for doc in preprocessed_documents:
            chunks.extend(self._chunker.chunk(doc))

        vectors = self._embedder.encode_documents([c.text for c in chunks])

        timestamp = datetime.now(timezone.utc)
        index_version = compute_index_version(
            self._embedder.model_name,
            self._chunker.config_signature,
            timestamp=timestamp,
        )

        collection_name = collection_name_for_version(
            self._settings.vector_store.qdrant.collection_name, index_version
        )
        self._vector_store.create_collection(collection_name, dimension=self._embedder.dimension)
        self._vector_store.upsert(collection_name, chunks, vectors)

        lexical_index = self._lexical_index_factory()
        lexical_index.build(chunks)
        lexical_path = lexical_path_for_version(
            Path(self._settings.admin.lexical_index_dir), index_version
        )
        lexical_index.save(lexical_path)

        manifest_path = Path(self._settings.admin.manifest_path)
        manifest = IndexManifest.load(manifest_path)
        manifest.record_new(
            {
                "index_version": index_version,
                "created_at": timestamp.isoformat(),
                "embedding_model": self._embedder.model_name,
                "embedding_dimension": self._embedder.dimension,
                "chunking_strategy": self._settings.chunking.strategy,
                "chunking_config_signature": self._chunker.config_signature,
                "lexical_lemmatization": self._settings.lexical.use_lemmatization,
                "bm25_params": {
                    "k1": self._settings.lexical.bm25.k1,
                    "b": self._settings.lexical.bm25.b,
                },
                "vector_collection_name": collection_name,
                "lexical_index_path": str(lexical_path),
                "document_count": len(preprocessed_documents),
                "chunk_count": len(chunks),
                "source_corpus": source_corpus,
            }
        )
        manifest.save(manifest_path)

        return index_version


def _content_type_of(document: Document) -> Literal["txt", "html", "plain"]:
    content_type = document.extra.get("content_type", "plain")
    if content_type not in ("txt", "html", "plain"):
        return "plain"
    return content_type
