"""Расширенные функциональные тесты edge-case для эндпоинта ``/search``.

Дополняют базовые сценарии из ``test_search_api.py`` граничными случаями:
пустой запрос, очень большой / нулевой ``top_k``, некорректный ``mode``,
поиск по непроиндексированному индексу, противоречивые и «пустые»
must_contain / must_exclude фильтры, очень длинный запрос и запрос с
юникодом/эмодзи.

Как и остальные integration-тесты, работают против реального embedded Qdrant
и реального dev ST-эмбеддера (фикстуры ``client`` / ``fresh_client`` из
``conftest.py``), поэтому помечены ``slow``.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.slow

SAMPLE_DOCS = [
    {
        "doc_id": "e1",
        "text": "Договор аренды нежилого помещения регулируется Гражданским кодексом РФ.",
        "source": "test",
    },
    {
        "doc_id": "e2",
        "text": "API — программный интерфейс для обмена данными по протоколу HTTP.",
        "source": "test",
    },
    {
        "doc_id": "e3",
        "text": "Сегодня в городе открылась новая выставка современного искусства.",
        "source": "test",
    },
    {
        "doc_id": "e4",
        "text": "REST и HTTP — основа большинства современных веб-сервисов и интеграций.",
        "source": "test",
    },
]

_HIT_KEYS = {"chunk_id", "doc_id", "text", "score", "metadata", "highlights"}


@pytest.fixture(scope="module", autouse=True)
def _index_edge_corpus(client):
    """Индексирует небольшой корпус один раз для всех тестов модуля."""

    response = client.post(
        "/index",
        json={"documents": SAMPLE_DOCS, "source_corpus": "search-edge"},
    )
    assert response.status_code == 200
    return response.json()


@pytest.mark.parametrize("mode", ["dense", "bm25", "hybrid", "hybrid_rerank"])
def test_empty_query_does_not_crash(client, mode):
    """Пустой запрос — не 5xx: сервис отрабатывает граациозно и возвращает
    корректную структуру ответа (список hits, возможно пустой)."""

    response = client.post("/search", json={"query": "", "mode": mode})

    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == mode
    assert isinstance(body["hits"], list)
    for hit in body["hits"]:
        assert set(hit.keys()) >= _HIT_KEYS


def test_very_large_top_k_is_capped_by_corpus_size(client):
    """Очень большой ``top_k`` не должен приводить к ошибке; число hits не
    может превышать число проиндексированных чанков (их не больше, чем
    документов * несколько чанков)."""

    response = client.post(
        "/search",
        json={"query": "договор", "mode": "hybrid", "top_k": 100_000},
    )

    assert response.status_code == 200
    hits = response.json()["hits"]
    # Корпус крошечный — hits физически ограничены числом чанков.
    assert len(hits) <= 100
    # doc_id не дублируются бесконечно — все из известного набора.
    assert {h["doc_id"] for h in hits} <= {d["doc_id"] for d in SAMPLE_DOCS}


def test_top_k_zero_returns_empty_hits(client):
    """``top_k=0`` — валидный граничный случай: retrieval отрабатывает, но
    выдача усечена до пустого списка."""

    response = client.post(
        "/search",
        json={"query": "договор аренды", "mode": "dense", "top_k": 0},
    )

    assert response.status_code == 200
    assert response.json()["hits"] == []


def test_invalid_mode_is_rejected_by_validation(client):
    """Некорректный ``mode`` отсекается Pydantic-валидацией (Literal) до
    попадания в сервис — HTTP 422, а не 500."""

    response = client.post(
        "/search",
        json={"query": "договор", "mode": "quantum"},
    )

    assert response.status_code == 422


def test_search_on_unindexed_index_returns_404(fresh_client):
    """Запрос к системе без активного индекса — 404 (NoActiveIndexError),
    а не падение."""

    response = fresh_client.post(
        "/search",
        json={"query": "что угодно", "mode": "hybrid_rerank"},
    )

    assert response.status_code == 404


@pytest.mark.parametrize("mode", ["dense", "bm25", "hybrid", "hybrid_rerank"])
def test_contradictory_must_contain_and_exclude_yields_no_hits(client, mode):
    """Один и тот же термин одновременно в must_contain и must_exclude —
    противоречие: строгий фильтр обязан вернуть пустую выдачу во всех
    режимах."""

    response = client.post(
        "/search",
        json={
            "query": "веб-сервисы",
            "mode": mode,
            "must_contain": ["REST"],
            "must_exclude": ["REST"],
            "top_k": 10,
        },
    )

    assert response.status_code == 200
    assert response.json()["hits"] == []


def test_must_contain_absent_term_yields_no_hits(client):
    """must_contain с термином, которого нет ни в одном документе, —
    выдача пуста, а не ошибка."""

    response = client.post(
        "/search",
        json={
            "query": "договор",
            "mode": "hybrid",
            "must_contain": ["блокчейн-квантовый-суперкомпьютер"],
            "top_k": 10,
        },
    )

    assert response.status_code == 200
    assert response.json()["hits"] == []


def test_very_long_query_is_handled(client):
    """Очень длинный запрос (тысячи символов) не должен ломать пайплайн."""

    long_query = "договор аренды нежилого помещения " * 500

    response = client.post(
        "/search",
        json={"query": long_query, "mode": "hybrid", "top_k": 5},
    )

    assert response.status_code == 200
    assert isinstance(response.json()["hits"], list)


@pytest.mark.parametrize("mode", ["dense", "bm25", "hybrid", "hybrid_rerank"])
def test_unicode_and_emoji_query_is_handled(client, mode):
    """Запрос с эмодзи и разным юникодом обрабатывается без ошибок во всех
    режимах (важно для русскоязычной системы: препроцессинг/токенизация не
    должны падать на непечатных/символьных данных)."""

    response = client.post(
        "/search",
        json={"query": "договор 😀 аренда 🏢 №№ ©", "mode": mode, "top_k": 5},
    )

    assert response.status_code == 200
    assert isinstance(response.json()["hits"], list)
