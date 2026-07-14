from __future__ import annotations

from sentence_transformers import SentenceTransformer

from app.core.types import Vector
from app.embedding.base import Embedder


class SentenceTransformerEmbedder(Embedder):
    """Wraps `sentence_transformers.SentenceTransformer`.

    One instance is meant to be shared (as a process-wide singleton, see
    `factory.get_or_build_embedder`) between indexing and query time so the
    vector space is guaranteed comparable (Ф2.5).
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        batch_size: int = 32,
        query_prefix: str = "",
        passage_prefix: str = "",
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._batch_size = batch_size
        self._query_prefix = query_prefix
        self._passage_prefix = passage_prefix
        self._model = SentenceTransformer(model_name, device=device)

    def encode_documents(self, texts: list[str]) -> list[Vector]:
        prefixed = [f"{self._passage_prefix}{t}" for t in texts]
        embeddings = self._model.encode(
            prefixed,
            batch_size=self._batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [vec.tolist() for vec in embeddings]

    def encode_query(self, text: str) -> Vector:
        prefixed = f"{self._query_prefix}{text}"
        embedding = self._model.encode(
            [prefixed],
            batch_size=self._batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return embedding[0].tolist()

    @property
    def dimension(self) -> int:
        return self._model.get_sentence_embedding_dimension()

    @property
    def model_name(self) -> str:
        return self._model_name
