"""Нагрузочные тесты (Locust) для эндпоинта ``POST /search``.

Проверяет пропускную способность (RPS), латентность (p50/p95/p99) и процент
ошибок под нагрузкой для всех четырёх режимов поиска: ``dense``, ``bm25``,
``hybrid`` и ``hybrid_rerank``. Каждый режим оформлен как отдельная задача
Locust с одинаковым весом, поэтому нагрузка распределяется по ним равномерно;
запросы выбираются случайно из набора реалистичных русскоязычных формулировок.

Предусловия (тест НЕ поднимает сервер и НЕ индексирует данные сам):

1. Сервер запущен и доступен по ``--host`` (по умолчанию ``http://localhost:8000``)::

       uvicorn app.main:app --host 0.0.0.0 --port 8000

2. Корпус уже проиндексирован, иначе ``/search`` вернёт 404
   (``NoActiveIndexError``). Достаточно демонстрационного корпуса из README
   или синтетического корпуса из ``tests/test_data/synthetic_corpus.json``::

       curl -X POST http://localhost:8000/index \
         -H "Content-Type: application/json" \
         --data-binary @- <<'JSON'
       {"documents": [{"doc_id": "1", "text": "Договор аренды регулируется ГК РФ.", "source": "demo"}],
        "source_corpus": "loadtest"}
       JSON

Запуск (headless, с выгрузкой метрик в CSV)::

    pip install locust
    locust -f tests/load/locustfile.py \
        --host http://localhost:8000 \
        --users 20 --spawn-rate 5 --run-time 2m \
        --headless --csv results_load/loadtest

Число одновременных пользователей и скорость их появления параметризуются
флагами ``--users`` / ``--spawn-rate``. После прогона ``--csv`` создаёт файлы
``*_stats.csv`` (в т.ч. столбцы p50/p95/p99 и Requests/s) и
``*_failures.csv`` (процент ошибок). Веб-интерфейс с графиками латентности
доступен, если запустить без ``--headless`` и открыть ``http://localhost:8089``.

Переменные окружения:

* ``SEM_SEARCH_LOAD_TOP_K`` — значение ``top_k`` в запросах (по умолчанию 10).
"""

from __future__ import annotations

import os
import random

from locust import HttpUser, between, task

# Реалистичные русскоязычные запросы, тематически совпадающие с синтетическим
# корпусом (юридические нормы, ИТ-термины, новости) — чтобы под нагрузкой
# ретриверы возвращали непустую выдачу, а не деградировали на «пустых» запросах.
QUERIES: list[str] = [
    "расторжение договора аренды",
    "прекращение трудовых отношений",
    "статья 614 ГК РФ арендная плата",
    "налоговый вычет при покупке недвижимости",
    "закон о защите прав потребителей",
    "досрочно освободить квартиру",
    "уволиться по собственному желанию",
    "регистрация прав на недвижимость",
    "REST API обмен данными по HTTP",
    "база данных SQL запросы",
    "Docker контейнеризация приложения",
    "микросервисная архитектура веб-сервисов",
    "JWT токен авторизация пользователя",
    "Git контроль версий репозиторий",
    "протокол HTTP мультиплексирование",
    "выставка современного искусства",
    "футбольный матч сборная России",
    "автоматизация юридического документооборота",
    "кадастровый учёт объектов недвижимости",
    "неустойка за просрочку платежа",
]

TOP_K = int(os.environ.get("SEM_SEARCH_LOAD_TOP_K", "10"))


class SearchUser(HttpUser):
    """Виртуальный пользователь, посылающий запросы на ``/search`` во всех
    четырёх режимах. Паузу между запросами имитируем через ``between`` для
    приближения к реальному пользовательскому поведению."""

    wait_time = between(0.5, 2.0)

    def _search(self, mode: str) -> None:
        # `name` группирует статистику по режиму, а не по конкретному запросу,
        # поэтому в отчёте Locust будет отдельная строка p50/p95/p99 на режим.
        payload = {
            "query": random.choice(QUERIES),
            "mode": mode,
            "top_k": TOP_K,
        }
        with self.client.post(
            "/search",
            json=payload,
            name=f"/search [{mode}]",
            catch_response=True,
        ) as response:
            if response.status_code != 200:
                response.failure(
                    f"HTTP {response.status_code} для режима {mode}: "
                    f"{response.text[:200]}"
                )
                return
            body = response.json()
            if "hits" not in body:
                response.failure(f"В ответе нет поля 'hits' (режим {mode})")
            else:
                response.success()

    @task
    def search_dense(self) -> None:
        self._search("dense")

    @task
    def search_bm25(self) -> None:
        self._search("bm25")

    @task
    def search_hybrid(self) -> None:
        self._search("hybrid")

    @task
    def search_hybrid_rerank(self) -> None:
        self._search("hybrid_rerank")
