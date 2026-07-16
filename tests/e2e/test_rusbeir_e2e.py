"""Сквозной (E2E) тест качества поиска на held-out срезе в стиле RusBeIR.

В отличие от unit/integration-тестов, которые дёргают отдельные сервисы, этот
тест работает ТОЛЬКО через публичный HTTP-API системы: индексирует held-out
корпус через ``POST /index`` и выполняет запросы через ``POST /search`` в
режиме ``hybrid_rerank``, после чего считает информационно-поисковые метрики
(NDCG@10 и Recall@10) по relevance judgments (qrels) и проверяет, что качество
не ниже разумного порога.

Held-out данные: реальный срез RusBeIR в CI недоступен (нет сети/лицензии на
загрузку бенчмарка), поэтому используется синтетический русскоязычный корпус с
qrels из ``tests/test_data/`` — тот же, что и в ``test_acceptance.py``, чтобы
метрики и стиль совпадали. При наличии реального среза его достаточно положить
в тот же формат (список документов + ``{"queries": {query: {doc_id: rel}}}``).

Грузит реальные dev-модели (эмбеддер + cross-encoder реранкер), поэтому помечен
``slow``. Запуск::

    pytest tests/e2e/test_rusbeir_e2e.py -v          # только E2E
    pytest -m slow                                    # все медленные тесты
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import AdminConfig, AppMeta, QdrantConfig, Settings, VectorStoreConfig
from app.main import create_app

pytestmark = [pytest.mark.slow, pytest.mark.e2e]

_DATA_DIR = Path(__file__).resolve().parent.parent / "test_data"
_CORPUS_PATH = _DATA_DIR / "synthetic_corpus.json"
_QRELS_PATH = _DATA_DIR / "synthetic_qrels.json"


def _load_corpus() -> list[dict]:
    with open(_CORPUS_PATH, encoding="utf-8") as f:
        return json.load(f)


def _load_qrels() -> dict[str, dict[str, int]]:
    with open(_QRELS_PATH, encoding="utf-8") as f:
        return json.load(f)["queries"]


def _ndcg_at_k(ranked_doc_ids: list[str], qrels: dict[str, int], k: int = 10) -> float:
    """Бинарный NDCG@k: документы вне qrels считаются нерелевантными."""
    gains = [1.0 if qrels.get(doc_id, 0) >= 1 else 0.0 for doc_id in ranked_doc_ids[:k]]
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    ideal = sorted((1.0 if rel >= 1 else 0.0 for rel in qrels.values()), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def _recall_at_k(ranked_doc_ids: list[str], qrels: dict[str, int], k: int = 10) -> float:
    """Recall@k по бинарной релевантности."""
    relevant = {doc_id for doc_id, rel in qrels.items() if rel >= 1}
    if not relevant:
        return 0.0
    retrieved = set(ranked_doc_ids[:k])
    return len(relevant & retrieved) / len(relevant)


def _make_settings(root: Path) -> Settings:
    return Settings(
        app=AppMeta(data_dir=str(root)),
        admin=AdminConfig(
            manifest_path=str(root / "index_manifest.json"),
            query_log_path=str(root / "logs" / "queries.jsonl"),
            lexical_index_dir=str(root / "lexical"),
        ),
        vector_store=VectorStoreConfig(
            qdrant=QdrantConfig(
                mode="embedded",
                path=str(root / "qdrant"),
                collection_name="e2e_chunks",
            )
        ),
    )


@pytest.fixture(scope="module")
def e2e_client(tmp_path_factory: pytest.TempPathFactory):
    """Module-scoped клиент над временным data-dir — модели грузятся один раз."""
    root = tmp_path_factory.mktemp("sem_search_e2e")
    settings = _make_settings(root)
    app = create_app(settings=settings)
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def _indexed(e2e_client: TestClient):
    """Индексирует held-out корпус через публичный ``/index`` один раз."""
    resp = e2e_client.post(
        "/index",
        json={"documents": _load_corpus(), "source_corpus": "rusbeir-e2e-heldout"},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


def _run_queries(client: TestClient, mode: str) -> tuple[float, float]:
    """Прогоняет все qrels-запросы через ``/search`` в режиме ``mode`` и
    возвращает средние (NDCG@10, Recall@10)."""
    queries = _load_qrels()
    ndcgs: list[float] = []
    recalls: list[float] = []
    for query_text, rel_map in queries.items():
        resp = client.post(
            "/search",
            json={"query": query_text, "mode": mode, "top_k": 10},
        )
        assert resp.status_code == 200, f"{mode} / {query_text!r}: {resp.text}"
        ranked = [hit["doc_id"] for hit in resp.json()["hits"]]
        ndcgs.append(_ndcg_at_k(ranked, rel_map, k=10))
        recalls.append(_recall_at_k(ranked, rel_map, k=10))
    assert ndcgs, "ожидался хотя бы один запрос в qrels"
    return sum(ndcgs) / len(ndcgs), sum(recalls) / len(recalls)


class TestRusBeIRE2E:
    """Сквозной прогон качества через публичный API."""

    def test_index_populated_whole_heldout_corpus(self, _indexed):
        """Санити-проверка: весь held-out корпус проиндексирован через /index."""
        assert _indexed["document_count"] == len(_load_corpus())
        assert _indexed["chunk_count"] >= _indexed["document_count"]
        assert _indexed["index_version"]

    def test_hybrid_rerank_quality_meets_threshold(self, e2e_client, _indexed):
        """NDCG@10 и Recall@10 гибрида с реранкингом не ниже фиксированного
        baseline на held-out срезе (аналогично порогам в test_acceptance.py)."""
        ndcg, recall = _run_queries(e2e_client, "hybrid_rerank")

        # Фиксированные пороги: на этом синтетическом срезе полноценный
        # пайплайн (hybrid + rerank) уверенно их проходит. Пороги намеренно
        # консервативны — они ловят регрессию (сломанный ретривер/реранкер
        # обрушит метрику к нулю), а не служат абсолютной оценкой качества
        # (реальные значения RusBeIR воспроизводятся на бенчмарке, глава 2 ВКР).
        assert ndcg >= 0.30, f"NDCG@10 hybrid_rerank слишком низкий: {ndcg:.4f}"
        assert recall >= 0.40, f"Recall@10 hybrid_rerank слишком низкий: {recall:.4f}"

    def test_hybrid_rerank_not_worse_than_dense_only(self, e2e_client, _indexed):
        """Ключевая E2E-проверка: полный пайплайн (hybrid + cross-encoder
        rerank) по среднему NDCG@10 не хуже чисто семантического (dense)
        поиска — с допуском на шум маленького синтетического среза."""
        dense_ndcg, _ = _run_queries(e2e_client, "dense")
        rerank_ndcg, _ = _run_queries(e2e_client, "hybrid_rerank")

        tolerance = 0.10  # тот же допуск, что и в test_acceptance.py
        assert rerank_ndcg >= dense_ndcg - tolerance, (
            f"hybrid_rerank NDCG@10 ({rerank_ndcg:.4f}) более чем на "
            f"{tolerance:.2f} ниже dense-only ({dense_ndcg:.4f})"
        )
