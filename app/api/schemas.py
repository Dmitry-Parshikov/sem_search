"""Pydantic request/response models for the API layer (Ф1.1-Ф1.6, Ф4.1-Ф4.3).

Kept separate from `app.core.types` on purpose: `Document` etc. are internal
frozen dataclasses used across every module, while these are the HTTP-facing
wire format (and may evolve independently, e.g. adding request validation
that has no business living in the domain type).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class IndexDocumentIn(BaseModel):
    doc_id: str
    text: str
    source: str = "api"
    section: str | None = None
    date: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class IndexRequest(BaseModel):
    documents: list[IndexDocumentIn]
    source_corpus: str = "default"


class IndexResponse(BaseModel):
    index_version: str
    document_count: int
    chunk_count: int


class ReindexRequest(BaseModel):
    source_corpus: str | None = None


class ReindexResponse(BaseModel):
    index_version: str
    document_count: int
    chunk_count: int
    source_corpus: str


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    index_version: str | None
    subsystems: dict[str, bool | str]


class SearchRequest(BaseModel):
    query: str
    mode: Literal["dense", "bm25", "hybrid", "dense_rerank", "hybrid_rerank"] | None = None
    top_k: int | None = None
    must_contain: list[str] = Field(default_factory=list)
    must_exclude: list[str] = Field(default_factory=list)


class SearchHitOut(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)
    highlights: list[str] = Field(default_factory=list)


class SearchResponse(BaseModel):
    hits: list[SearchHitOut]
    index_version: str
    mode: str
    query: str
    # Ф2.2: suggested correction for `query` (never used for retrieval itself).
    typo_suggestion: str | None = None
    # Ф2.4: the expanded query actually used for retrieval, only set when it
    # differs from `query`.
    expanded_query: str | None = None
    # NFR "Надёжность": non-empty when an optional query-processing stage
    # (typo correction, term expansion) failed and was skipped gracefully.
    warnings: list[str] = Field(default_factory=list)


class VersionInfo(BaseModel):
    """Typed mirror of one `IndexManifest` version entry (see
    `app.admin.manifest.IndexManifest`'s docstring for the authoritative
    field list) -- used for the `/admin/versions` API response instead of
    exposing the bare manifest dict directly."""

    index_version: str
    created_at: str
    embedding_model: str
    embedding_dimension: int
    chunking_strategy: str
    chunking_config_signature: str
    lexical_lemmatization: bool
    bm25_params: dict[str, float]
    vector_collection_name: str
    lexical_index_path: str
    document_count: int
    chunk_count: int
    source_corpus: str
    status: Literal["active", "superseded"]


class VersionsResponse(BaseModel):
    versions: list[VersionInfo]
    active_version: str | None = None


class RollbackResponse(BaseModel):
    active_version: str
    created_at: str
    document_count: int
    chunk_count: int
    source_corpus: str


class FolderIndexRequest(BaseModel):
    """POST /index-from-folder: path to a local folder with documents."""

    folder_path: str = Field(..., description="Absolute path to a folder with documents")
    source_label: str = Field("folder", description="Source corpus label for versioning")
    mode: str = Field("replace", description="'replace' — full replacement; 'append' — add to existing corpus")


class FileError(BaseModel):
    path: str
    suffix: str | None = None
    error: str


class FolderIndexResponse(BaseModel):
    index_version: str
    documents_found: int
    documents_indexed: int
    chunk_count: int
    errors: list[FileError] = Field(default_factory=list)


# ── Admin config response (GET /admin/config) ──────────────────────────

class EmbeddingConfigOut(BaseModel):
    model_name: str
    device: str
    batch_size: int
    query_prefix: str
    passage_prefix: str


class ChunkingConfigOut(BaseModel):
    strategy: str
    params: dict[str, Any]


class RerankingConfigOut(BaseModel):
    enabled: bool
    model_name: str
    device: str
    top_n: int
    batch_size: int


class HybridizationConfigOut(BaseModel):
    method: str
    rrf_k: int


class TypoCorrectionConfigOut(BaseModel):
    enabled: bool
    max_distance: int
    score_cutoff: float


class QueryProcessingConfigOut(BaseModel):
    typo_correction: TypoCorrectionConfigOut
    dictionaries_enabled: bool
    dictionaries_dir: str


class SearchConfigOut(BaseModel):
    default_mode: str
    default_top_k: int


class AdminConfigResponse(BaseModel):
    config_profile: str
    embedding: EmbeddingConfigOut
    chunking: ChunkingConfigOut
    reranking: RerankingConfigOut
    hybridization: HybridizationConfigOut
    query_processing: QueryProcessingConfigOut
    search: SearchConfigOut


class CorpusInfo(BaseModel):
    name: str
    document_count: int
    size_bytes: int
    last_modified: str


class CorporaResponse(BaseModel):
    corpora: list[CorpusInfo]


class CorpusDocumentInfo(BaseModel):
    doc_id: str
    text_preview: str
    text_length: int


class CorpusDetailResponse(BaseModel):
    name: str
    document_count: int
    documents: list[CorpusDocumentInfo]


class DeleteDocumentResponse(BaseModel):
    deleted: bool
    doc_id: str
    corpus: str
    message: str = ""


# ── Dictionaries API ────────────────────────────────────────────────

class DictionaryInfo(BaseModel):
    filename: str
    entry_count: int
    size_bytes: int


class DictionariesResponse(BaseModel):
    enabled: bool
    dictionaries: list[DictionaryInfo]


class DictionaryToggleRequest(BaseModel):
    enabled: bool
