from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from app.admin.versioning import (
    collection_name_for_version,
    compute_index_version,
    lexical_path_for_version,
)


def test_compute_index_version_deterministic_given_fixed_timestamp():
    ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
    v1 = compute_index_version("model-a", "sig-a", timestamp=ts)
    v2 = compute_index_version("model-a", "sig-a", timestamp=ts)
    assert v1 == v2


def test_compute_index_version_differs_when_model_changes():
    ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
    v1 = compute_index_version("model-a", "sig-a", timestamp=ts)
    v2 = compute_index_version("model-b", "sig-a", timestamp=ts)
    assert v1 != v2


def test_compute_index_version_differs_when_chunking_signature_changes():
    ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
    v1 = compute_index_version("model-a", "sig-a", timestamp=ts)
    v2 = compute_index_version("model-a", "sig-b", timestamp=ts)
    assert v1 != v2


def test_compute_index_version_hash_suffix_stable_across_timestamps():
    """The hash suffix (last 6 hex chars) depends only on model+chunking
    signature, not on the timestamp -- only the human-readable prefix
    should differ."""

    ts1 = datetime(2024, 1, 1, tzinfo=timezone.utc)
    ts2 = datetime(2025, 6, 15, tzinfo=timezone.utc)
    v1 = compute_index_version("model-a", "sig-a", timestamp=ts1)
    v2 = compute_index_version("model-a", "sig-a", timestamp=ts2)
    assert v1 != v2  # timestamps differ, so full strings differ
    assert v1.split("_")[-1] == v2.split("_")[-1]  # but hash suffix matches


def test_collection_name_for_version_format():
    assert collection_name_for_version("sem_search_chunks", "v_20240101T000000Z_abc123") == (
        "sem_search_chunks__v_20240101T000000Z_abc123"
    )


def test_lexical_path_for_version_format():
    path = lexical_path_for_version(Path("./data/lexical"), "v_20240101T000000Z_abc123")
    assert path == Path("./data/lexical/v_20240101T000000Z_abc123.pkl")
