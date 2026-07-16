# Тесты sem_search

Тестовый набор разбит на четыре категории по назначению и стоимости запуска.

| Категория | Путь | Маркер | Модели | Что проверяет |
|-----------|------|--------|--------|---------------|
| Unit | `tests/unit/` | — | нет | Отдельные модули изолированно (чанкинг, токенизация, RRF, фильтры, версионирование и т.д.) |
| Integration | `tests/integration/` | `slow` | реальные dev-модели | API-слой через FastAPI `TestClient`: `/index`, `/search`, `/reindex`, `/health`, `/admin`, edge-cases |
| E2E | `tests/e2e/` | `slow`, `e2e` | реальные dev-модели | Сквозное качество только через публичный HTTP-API: `/index` → `/search` + метрики NDCG@10 / Recall@10 |
| Load | `tests/load/` | — (Locust, не pytest) | — (нужен запущенный сервер) | Нагрузка на `/search`: RPS, латентность p50/p95/p99, процент ошибок |

Быстрые unit-тесты не грузят ML-модели и выполняются за секунды. Всё, что
помечено `slow` (integration + e2e), поднимает реальный dev-эмбеддер
(`intfloat/multilingual-e5-small`) и cross-encoder реранкер — первый запуск
скачивает модели из HuggingFace Hub.

> DEV vs FINAL модели: по умолчанию используются облегчённые dev-модели. В
> экспериментах ВКР зафиксированы `deepvk/USER2-base` (эмбеддер) и
> `BAAI/bge-reranker-v2-m3` (реранкер) — переключаются через `config/config.yaml`
> или env-переменные (см. корневой `README.md`).

## Как запускать

### Быстрые тесты (без моделей)

```bash
pytest -m "not slow"
```

### Все тесты (unit + integration + e2e), включая медленные

```bash
pytest
```

### По категориям

```bash
pytest tests/unit/                       # только unit
pytest tests/integration/                # только integration (slow)
pytest tests/e2e/                         # только e2e (slow)
pytest -m e2e                             # e2e через маркер
pytest -m slow                            # все медленные (integration + e2e)
```

## Покрытие (pytest-cov)

`pytest-cov` включён в `requirements.txt`. Покрытие пакета `app`:

```bash
# Покрытие по быстрым тестам (term + HTML отчёт в htmlcov/)
pytest --cov=app --cov-report=term-missing --cov-report=html -m "not slow"

# Полное покрытие по всем тестам (грузит модели, дольше)
pytest --cov=app --cov-report=term-missing --cov-report=html
```

`--cov-report=term-missing` печатает непокрытые строки прямо в консоль,
`--cov-report=html` создаёт браузируемый отчёт в `htmlcov/index.html`.
Конкретный целевой процент покрытия не задан — инструментарий предназначен для
навигации по непокрытым участкам.

## Нагрузочные тесты (Locust)

Живут отдельно от pytest — это сценарий Locust (`tests/load/locustfile.py`),
который бьёт по `/search` во всех четырёх режимах (`dense`, `bm25`, `hybrid`,
`hybrid_rerank`) реалистичными русскоязычными запросами.

**Предусловия** (тест НЕ поднимает сервер и НЕ индексирует данные):

1. Запущенный сервер:

   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```

2. Проиндексированный корпус (иначе `/search` вернёт 404). Достаточно
   демо-корпуса из корневого `README.md` или синтетического корпуса из
   `tests/test_data/synthetic_corpus.json`.

**Запуск** (headless, метрики в CSV):

```bash
pip install locust
locust -f tests/load/locustfile.py \
    --host http://localhost:8000 \
    --users 20 --spawn-rate 5 --run-time 2m \
    --headless --csv results_load/loadtest
```

После прогона `--csv` создаёт `results_load/loadtest_stats.csv` (столбцы
Requests/s и перцентили латентности p50/p95/p99) и
`results_load/loadtest_failures.csv` (процент ошибок). Число пользователей и
скорость их появления задаются `--users` / `--spawn-rate`. Без `--headless`
доступен веб-интерфейс с графиками на `http://localhost:8089`.

## Тестовые данные

`tests/test_data/` — синтетический русскоязычный корпус (33 документа: юр.
нормы, ИТ-термины, новости + смешанные) с бинарными relevance judgments по 20
запросам. Используется приёмочными (`integration/test_acceptance.py`) и E2E
(`e2e/test_rusbeir_e2e.py`) тестами для расчёта NDCG@10 / Recall@10. Реальный
срез RusBeIR в CI недоступен, поэтому синтетика служит held-out срезом того же
формата.
