from __future__ import annotations

import hashlib
from datetime import datetime, timezone


def make_chunk_id(doc_id: str, position: int) -> str:
    return f"{doc_id}::{position:04d}"


def compute_index_version(model_name: str, chunking_signature: str, timestamp: datetime | None = None) -> str:
    """index_version = v_{timestamp}_{hash(model + chunking_signature)[:6]}.

    The timestamp lives in the human-readable prefix (so versions sort
    lexicographically by creation time) while the hash suffix ties the
    version to the exact model/chunking config that produced it.
    """
    ts = timestamp or datetime.now(timezone.utc)
    # Microsecond precision (not just seconds) so two runs issued in quick
    # succession -- e.g. index immediately followed by reindex -- still get
    # distinct versions, hence distinct Qdrant collections/lexical pickles.
    ts_str = ts.strftime("%Y%m%dT%H%M%S%fZ")
    digest = hashlib.sha256(f"{model_name}|{chunking_signature}".encode("utf-8")).hexdigest()[:6]
    return f"v_{ts_str}_{digest}"
