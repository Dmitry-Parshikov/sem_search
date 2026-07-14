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
