# sem_search — семантическая поисковая система для русскоязычных текстов

Прототип семантической поисковой системы, реализованный в рамках выпускной
квалификационной работы (ВКР) на тему «Разработка семантической поисковой
системы для русскоязычных текстов».

**Возможности:** полнотекстовый поиск по коллекциям русскоязычных документов
в трёх режимах — лексический (BM25), семантический (dense embeddings) и
гибридный (RRF fusion) — с опциональным переранжированием (cross-encoder),
исправлением опечаток, расширением запроса словарём терминов и
пост-фильтрацией (must-contain / must-exclude).

---

## Быстрый старт (< 5 минут)

```bash
# 1. Клонировать и установить зависимости
git clone <repo-url> sem_search
cd sem_search
python -m venv .venv
source .venv/bin/activate    # или .venv\Scripts\activate на Windows
pip install -r requirements.txt

# 2. Запустить сервер (embedded Qdrant, Docker не нужен)
uvicorn app.main:app --reload

# 3. Проиндексировать демонстрационный корпус
curl -X POST http://localhost:8000/index \
  -H "Content-Type: application/json" \
  -d '{"documents": [
    {"doc_id": "1", "text": "Договор аренды регулируется ГК РФ.", "source": "demo"},
    {"doc_id": "2", "text": "REST API — основа современных веб-сервисов.", "source": "demo"},
    {"doc_id": "3", "text": "Сегодня в городе открылась выставка искусства.", "source": "demo"}
  ], "source_corpus": "demo"}'

# 4. Выполнить поиск
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "аренда", "mode": "hybrid_rerank", "top_k": 5}'

# 5. Проверить статус
curl http://localhost:8000/health
```

---

## Запуск через Docker

```bash
# Продакшен-режим: app + Qdrant в отдельных контейнерах
docker compose -f docker/docker-compose.yml up --build

# Standalone (embedded Qdrant внутри контейнера):
docker build -t sem_search -f docker/Dockerfile .
docker run -p 8000:8000 -v ./data:/app/data sem_search
```

После запуска сервис доступен на `http://localhost:8000`, интерактивная
документация API (Swagger UI) — на `http://localhost:8000/docs`.

---

## Архитектура

```
ИНДЕКСАЦИЯ (offline)                     ОБРАБОТКА ЗАПРОСА (online)
Документы → Препроцессинг → Чанкинг       Запрос → исправление опечаток
→ Эмбеддер → { Векторное хранилище,        → расширение словарём терминов
               Лексический индекс }         → разбор must-contain/must-exclude
                                           → Эмбеддер (тот же инстанс)
                                           → { Dense-ретривер, BM25-ретривер }
                                           → Гибридизация (RRF/weighted)
                                           → Строгий лексический фильтр
                                           → Переранжирование (cross-encoder)
                                           → Ранжированная выдача
```

Каждый модуль реализован за абстрактным интерфейсом (ABC) — замена эмбеддера
или векторного хранилища требует правки только в конфиге, не в коде.

### Структура проекта

```
sem_search/
├── app/
│   ├── api/                 # FastAPI роуты: index, search, reindex, health, admin
│   ├── admin/               # версионирование индекса, манифест, логирование запросов
│   ├── chunking/            # стратегии чанкинга: fixed_window, sentence_window, paragraph
│   ├── config.py            # pydantic-settings + загрузка config.yaml
│   ├── core/                # общие типы, ошибки
│   ├── embedding/           # обёртка над sentence-transformers
│   ├── hybrid/              # RRF и weighted fusion
│   ├── indexing/            # сервис индексации
│   ├── lexical/             # BM25-индекс + русская лемматизация (pymorphy3)
│   ├── main.py              # точка входа FastAPI
│   ├── preprocessing/       # очистка текста, юникод-нормализация
│   ├── query/               # исправление опечаток (rapidfuzz), расширение словарём
│   ├── rerank/              # cross-encoder reranker
│   ├── search/              # сервис поиска, ретриверы, фильтры
│   └── vector_store/        # абстракция + Qdrant (embedded / remote)
├── config/
│   ├── config.yaml          # основной конфиг (embedded Qdrant)
│   ├── docker.yaml          # конфиг для Docker (remote Qdrant)
│   └── terms_dictionary.yaml # словарь аббревиатур/синонимов
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── tests/
│   ├── unit/                # 127 быстрых юнит-тестов
│   ├── integration/         # 64 интеграционных теста (медленных, с ML-моделями)
│   └── test_data/           # синтетический корпус + relevance judgments
├── requirements.txt
└── README.md
```

---

## API эндпоинты

| Метод | Эндпоинт | Назначение |
|-------|----------|------------|
| `GET` | `/health` | Статус подсистем (vector store, embedder, reranker) + текущая версия индекса |
| `POST` | `/index` | Индексация документов: препроцессинг → чанкинг → векторизация → запись |
| `POST` | `/index-from-folder` | Пакетная индексация из локальной папки (.txt, .md, .docx, .rtf) |
| `POST` | `/search` | Поиск: `query`, `mode`, `top_k`, `must_contain`, `must_exclude` |
| `POST` | `/reindex` | Полная переиндексация последнего проиндексированного корпуса |
| `GET` | `/admin/versions` | Список всех версий индекса со статусами |
| `POST` | `/admin/rollback/{version}` | Откат активного индекса к указанной версии |

### Режимы поиска (`mode`)

| Режим | Описание |
|-------|----------|
| `dense` | Чисто семантический поиск (ANN по косинусному сходству) |
| `bm25` | Чисто лексический поиск (BM25 + опциональная лемматизация) |
| `hybrid` | Гибридный (RRF-слияние dense + BM25) + пост-фильтрация |
| `hybrid_rerank` | Гибридный + переранжирование топ-N кросс-энкодером (по умолчанию) |

### Примеры запросов

```bash
# Семантический поиск
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "как расторгнуть договор", "mode": "dense", "top_k": 5}'

# Гибридный поиск с обязательными и исключаемыми словами
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "веб-сервисы", "mode": "hybrid_rerank", "must_contain": ["REST"], "must_exclude": ["SOAP"]}'

# Индексация документов
curl -X POST http://localhost:8000/index \
  -H "Content-Type: application/json" \
  -d '{"documents": [{"doc_id": "1", "text": "...", "source": "legal"}], "source_corpus": "my-corpus"}'

# Переиндексация последнего корпуса
curl -X POST http://localhost:8000/reindex -H "Content-Type: application/json" -d '{}'

# Просмотр версий индекса
curl http://localhost:8000/admin/versions

# Откат к предыдущей версии
curl -X POST http://localhost:8000/admin/rollback/v_20240101T120000Z_abc123
```

---

## Конфигурация

Основной конфиг — `config/config.yaml`. Все параметры могут быть переопределены
через переменные окружения с префиксом `SEM_SEARCH_` и двойным подчёркиванием
в качестве разделителя вложенных полей:

```bash
SEM_SEARCH_EMBEDDING__MODEL_NAME="BAAI/bge-m3"
SEM_SEARCH_VECTOR_STORE__QDRANT__MODE="remote"
SEM_SEARCH_VECTOR_STORE__QDRANT__URL="http://qdrant:6333"
```

### DEV vs FINAL профиль

По умолчанию используются облегчённые модели для быстрой разработки и тестов:

| Компонент | DEV-модель | FINAL-модель |
|-----------|-----------|-------------|
| Embedder | `intfloat/multilingual-e5-small` | `BAAI/bge-m3` |
| Reranker | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` | `BAAI/bge-reranker-v2-m3` |

Для переключения на финальные модели измените `model_name` в `config/config.yaml`
или установите переменные окружения.

### Словарь терминов

`config/terms_dictionary.yaml` — YAML-словарь вида `"аббревиатура": ["раскрытие", ...]`.
Подключается и редактируется без изменения кода. Пример:

```yaml
бд:
  - база данных
АПК:
  - Арбитражный процессуальный кодекс
  - аппаратно-программный комплекс
```

---

## Тестирование

```bash
# Быстрые тесты (без ML-моделей, ~5 сек)
pytest tests/ -m "not slow"

# Все тесты, включая интеграционные (~30-60 сек, загружают реальные модели)
pytest tests/

# Только юнит-тесты
pytest tests/unit/

# Только интеграционные
pytest tests/integration/

# Приёмочные тесты (критерии раздела 10 ТЗ)
pytest tests/integration/test_acceptance.py -v
```

**Покрытие:** 191 тест (127 быстрых + 64 медленных), включая 16 приёмочных
тестов по всем 6 критериям раздела 10 технического задания.

### Синтетический тестовый корпус

Тестовые данные в `tests/test_data/` — **синтетический** корпус из 33 документов
на русском языке, созданный исключительно для демонстрации и автотестов, а не
в результате сбора реальных данных. Охватывает три тематики (юридические нормы,
ИТ-термины, общие новости) плюс смешанные документы. Размечен 20 тестовыми
запросами с бинарными relevance judgments для расчёта NDCG@10.

---

## Выбор технологий и обоснование

| Компонент | Выбор | Обоснование |
|-----------|-------|------------|
| **Embedder** | `intfloat/multilingual-e5-small` (DEV) / `BAAI/bge-m3` (FINAL) | Поддержка русского, размерность 384/1024, хорошие результаты на RusBeIR |
| **Reranker** | `mmarco-mMiniLMv2` (DEV) / `bge-reranker-v2-m3` (FINAL) | Cross-encoder, совместная обработка пары «запрос-документ», поддержка русского в final-модели |
| **BM25** | `rank-bm25` | Чистый Python, простая интеграция, обёрнут в абстрактный `LexicalIndex` |
| **Векторное хранилище** | `Qdrant` (embedded / remote) | HNSW-based ANN, встроенная фильтрация по payload, два режима работы |
| **Лемматизация** | `pymorphy3` | Активно поддерживаемый форк `pymorphy2` для русского, совместим с Python 3.11+ |
| **Опечатки** | `rapidfuzz` (Левенштейн) | Быстрая C++ реализация, поверх словаря термов из индекса |
| **Конфигурация** | `pydantic-settings` + YAML | Валидация типов, переопределение через env vars, единый файл |
| **API** | FastAPI + Uvicorn | Автоматическая OpenAPI-документация, валидация через Pydantic, async |
| **Логирование** | `structlog` | Структурированные логи, читаемые человеком и машиной |

### Почему Qdrant, а не FAISS?

FAISS не поддерживает фильтрацию по метаданным на уровне индекса — пришлось бы
реализовывать её вручную поверх результатов ANN-поиска. Qdrant имеет встроенную
фильтрацию по payload, два режима работы (embedded для dev, remote для production)
и единый API для обоих — что позволяет переключаться без изменения кода.

---

## Ограничения (соответствуют разделу 11 ТЗ — ВКР)

1. **Прототип.** Это учебный прототип, а не production-система: нет
   горизонтального масштабирования, мониторинга, CI/CD.
2. **Синтетические данные.** Тестовый корпус создан искусственно для
   демонстрации и не отражает свойств реальных коллекций.
3. **Один инстанс модели.** Embedder и reranker загружаются в память один раз;
   нет шардирования или распределённого инференса.
4. **Русский язык.** Основной фокус на русском; для других языков потребуется
   замена лемматизатора, словаря терминов и, возможно, моделей.
5. **Полная переиндексация.** `/reindex` перестраивает индекс целиком;
   инкрементальное обновление не поддерживается.
6. **Метрики качества.** NDCG-оценки на синтетическом корпусе используются
   только для проверки относительного порядка (hybrid ≥ baseline).
   Абсолютные значения (65,71–65,85 hybrid vs 52,16 BM25 на RusBeIR)
   воспроизводятся только на реальном бенчмарке (глава 2 ВКР).

---

## Лицензия

Учебный проект. Код распространяется под лицензией MIT.
