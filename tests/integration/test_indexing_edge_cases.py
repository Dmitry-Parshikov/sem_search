"""Расширенные функциональные тесты edge-case для эндпоинта ``/index``.

Дополняют базовые сценарии из ``test_indexing_api.py``: дубликаты ``doc_id``,
документ с пустым текстом, юникод/эмодзи в тексте и очень большой одиночный
документ. Проверяют, что индексация не падает и что результат затем доступен
для поиска.

Работают против реального embedded Qdrant и dev ST-эмбеддера
(``fresh_client`` из ``conftest.py`` — по чистому индексу на тест), поэтому
помечены ``slow``.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.slow


def test_duplicate_doc_ids_are_all_indexed(fresh_client):
    """Дубликаты ``doc_id`` в одном запросе не приводят к ошибке: оба
    документа проходят пайплайн и учитываются в ``document_count``
    (система не выполняет дедупликацию на уровне индексации)."""

    docs = [
        {"doc_id": "dup", "text": "Договор аренды помещения регулируется ГК РФ.", "source": "test"},
        {"doc_id": "dup", "text": "REST API — основа современных веб-сервисов.", "source": "test"},
    ]

    response = fresh_client.post(
        "/index",
        json={"documents": docs, "source_corpus": "dup-corpus"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["document_count"] == 2
    assert body["chunk_count"] >= 2
    assert body["index_version"]

    # Индекс работоспособен: поиск по нему возвращает валидный ответ.
    search = fresh_client.post(
        "/search",
        json={"query": "договор аренды", "mode": "hybrid", "top_k": 10},
    )
    assert search.status_code == 200
    assert all(hit["doc_id"] == "dup" for hit in search.json()["hits"])


def test_empty_text_document_does_not_crash_indexing(fresh_client):
    """Документ с пустым текстом рядом с обычным документом: индексация
    отрабатывает, оба документа учтены, чанки создаются как минимум для
    непустого документа."""

    docs = [
        {"doc_id": "empty", "text": "", "source": "test"},
        {"doc_id": "normal", "text": "Гражданский кодекс регулирует договор аренды.", "source": "test"},
    ]

    response = fresh_client.post(
        "/index",
        json={"documents": docs, "source_corpus": "empty-text-corpus"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["document_count"] == 2
    assert body["chunk_count"] >= 1


def test_unicode_and_emoji_document_text_is_indexed(fresh_client):
    """Юникод и эмодзи в тексте документа не ломают препроцессинг/чанкинг."""

    docs = [
        {
            "doc_id": "uni",
            "text": "Выставка 🎨 современного искусства © 2026 №42 — вход свободный.",
            "source": "test",
        },
    ]

    response = fresh_client.post(
        "/index",
        json={"documents": docs, "source_corpus": "unicode-corpus"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["document_count"] == 1
    assert body["chunk_count"] >= 1


def test_large_single_document_is_indexed(fresh_client):
    """Очень большой одиночный документ разбивается на несколько чанков и
    индексируется без ошибок."""

    big_text = (
        "Договор аренды нежилого помещения заключается в письменной форме. "
        "Стороны обязаны указать срок и размер арендной платы. "
    ) * 300

    response = fresh_client.post(
        "/index",
        json={
            "documents": [{"doc_id": "big", "text": big_text, "source": "test"}],
            "source_corpus": "big-doc-corpus",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["document_count"] == 1
    # Большой текст обязан породить более одного чанка.
    assert body["chunk_count"] > 1
