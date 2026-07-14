# Инструкция по запуску и проверке поисковой системы sem_search

## 1. Установка и запуск

```powershell
# Клонировать репозиторий
git clone <repo-url> sem_search
cd sem_search

# Создать виртуальное окружение
python -m venv .venv
.venv\Scripts\activate

# Установить зависимости
pip install -r requirements.txt

# Запустить сервер
uvicorn app.main:app --reload
```

Сервер поднят на `http://localhost:8000`. Открой `http://localhost:8000/docs` — там Swagger UI со всеми эндпоинтами и формами для запросов.

---

## 2. Индексация документов

```powershell
# Простейшая индексация — 3 документа на разные темы
Invoke-RestMethod -Uri http://localhost:8000/index -Method Post -ContentType "application/json" -Body '{
  "documents": [
    {"doc_id": "law1", "text": "Договор аренды нежилого помещения регулируется Гражданским кодексом РФ. Расторжение договора возможно по соглашению сторон или через суд.", "source": "legal"},
    {"doc_id": "it1", "text": "REST API — архитектурный стиль взаимодействия клиента и сервера через HTTP. JSON Web Token используется для аутентификации запросов.", "source": "it"},
    {"doc_id": "news1", "text": "В центральном выставочном зале города открылась экспозиция современного искусства. Посетители могут приобрести картины напрямую у авторов.", "source": "news"},
    {"doc_id": "it2", "text": "Docker-контейнер — это изолированное окружение, содержащее приложение и все его зависимости. Образ собирается по Dockerfile однократно.", "source": "it"},
    {"doc_id": "law2", "text": "Наниматель вправе прекратить пользование жилым помещением, направив письменное уведомление за один месяц до выезда.", "source": "legal"}
  ],
  "source_corpus": "demo"
}'
```

Ответ покажет `index_version` — запиши его, пригодится.

---

## 3. Четыре режима поиска

### 3.1 Dense (семантический) — ищет по смыслу, а не по точным словам

```powershell
# Запрос «прекратить аренду» — найдёт law2 про «прекратить пользование помещением»,
# даже несмотря на то что слово «аренда» там не встречается ни разу
Invoke-RestMethod -Uri http://localhost:8000/search -Method Post -ContentType "application/json" -Body '{
  "query": "прекратить аренду жилья",
  "mode": "dense",
  "top_k": 5
}'
```

**Что смотреть:** в ответе поле `mode: "dense"`, результаты отсортированы по `score`, документы с синонимами («прекратить пользование» ≈ «прекратить аренду») в топе.

### 3.2 BM25 (лексический) — точное совпадение слов

```powershell
# «REST API» — точное совпадение, найдет it1
Invoke-RestMethod -Uri http://localhost:8000/search -Method Post -ContentType "application/json" -Body '{
  "query": "REST API HTTP",
  "mode": "bm25",
  "top_k": 5
}'
```

**Что смотреть:** результаты совпадают по конкретным словам, а не по смыслу. Если в документе нет слов из запроса — он не попадёт в выдачу.

### 3.3 Hybrid (RRF-слияние) — объединяет dense + BM25

```powershell
Invoke-RestMethod -Uri http://localhost:8000/search -Method Post -ContentType "application/json" -Body '{
  "query": "контейнер для приложения",
  "mode": "hybrid",
  "top_k": 5
}'
```

**Что смотреть:** результат — слияние двух списков (dense + BM25) через Reciprocal Rank Fusion. В выдаче есть как семантически близкие («Docker-контейнер»), так и лексически совпадающие документы.

### 3.4 Hybrid + Rerank (с переранжированием) — режим по умолчанию

```powershell
Invoke-RestMethod -Uri http://localhost:8000/search -Method Post -ContentType "application/json" -Body '{
  "query": "как расторгнуть договор аренды",
  "mode": "hybrid_rerank",
  "top_k": 5
}'
```

**Что смотреть:** порядок результатов отличается от `hybrid` — кросс-энкодер переоценил каждую пару «запрос-документ» и пересортировал топ.

---

## 4. Фильтрация: must_contain и must_exclude

```powershell
# Только документы, где ВСЕ слова из must_contain присутствуют
Invoke-RestMethod -Uri http://localhost:8000/search -Method Post -ContentType "application/json" -Body '{
  "query": "правовые вопросы",
  "mode": "hybrid_rerank",
  "must_contain": ["Гражданский"],
  "top_k": 5
}'

# Исключить документы, содержащие определённые слова
Invoke-RestMethod -Uri http://localhost:8000/search -Method Post -ContentType "application/json" -Body '{
  "query": "аренда",
  "mode": "hybrid_rerank",
  "must_exclude": ["нежилого"],
  "top_k": 5
}'
```

---

## 5. Исправление опечаток (typo correction)

```powershell
# Намеренная опечатка: «дагавор» вместо «договор»
Invoke-RestMethod -Uri http://localhost:8000/search -Method Post -ContentType "application/json" -Body '{
  "query": "дагавор аренды",
  "mode": "dense",
  "top_k": 5
}'
```

**Что смотреть:** в ответе есть поле `typo_suggestion` — система предлагает исправление (`"договор аренды"`), но не блокирует исходный запрос. Поиск выполняется по исправленной версии.

---

## 6. Расширение словарём терминов (term expansion)

Словарь лежит в `config/terms_dictionary.yaml`. Добавь туда свою аббревиатуру:

```yaml
# Было:
бд:
  - база данных

# Добавь:
жп:
  - жилое помещение
```

```powershell
# Теперь запрос с аббревиатурой будет расширен
Invoke-RestMethod -Uri http://localhost:8000/search -Method Post -ContentType "application/json" -Body '{
  "query": "аренда жп",
  "mode": "bm25",
  "top_k": 5
}'
```

**Что смотреть:** в ответе поле `expanded_query` содержит `"аренда жп жилое помещение"` — словарь раскрыл аббревиатуру, и поиск выполнен по расширенному запросу.

---

## 7. Администрирование: версии индекса и откат

### 7.1 Посмотреть все версии

```powershell
Invoke-RestMethod -Uri http://localhost:8000/admin/versions
```

Показывает: `active_version`, список всех версий со статусами (`active` / `superseded`), количество документов и чанков в каждой.

### 7.2 Переиндексация (создать новую версию)

```powershell
Invoke-RestMethod -Uri http://localhost:8000/reindex -Method Post -ContentType "application/json" -Body '{}'
```

Создаёт новую версию индекса из того же корпуса. Предыдущая активная версия становится `superseded`.

### 7.3 Откат к предыдущей версии

```powershell
# Возьми version из вывода GET /admin/versions (например, v_20240101T120000Z_abc123)
Invoke-RestMethod -Uri http://localhost:8000/admin/rollback/v_20240101T120000Z_abc123 -Method Post
```

После отката `/search` сразу начинает использовать старую версию индекса — перезапуск не нужен.

### 7.4 Логи запросов

```powershell
# Каждый поисковый запрос пишет одну JSON-строку в файл
Get-Content .\data\logs\queries.jsonl | Select-Object -Last 5
```

Каждая строка — JSON с полями: `query`, `mode`, `top_k`, `must_contain`, `must_exclude`, `index_version`, `response_time_ms`, `warnings`.

---

## 8. Проверка здоровья

```powershell
Invoke-RestMethod -Uri http://localhost:8000/health
```

Ответ:
```json
{
  "status": "ok",
  "index_version": "v_20260714T...",
  "subsystems": {
    "vector_store": true,
    "embedder": true,
    "reranker": true
  }
}
```

---

## 9. Быстрая проверка через curl (если PowerShell неудобен)

Те же запросы одной строкой через Git Bash (есть в VS Code):

```bash
# Индексация
curl -s -X POST http://localhost:8000/index \
  -H "Content-Type: application/json" \
  -d '{"documents":[{"doc_id":"1","text":"Договор аренды","source":"t"}],"source_corpus":"demo"}' | python -m json.tool

# Dense-поиск
curl -s -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query":"аренда","mode":"dense","top_k":3}' | python -m json.tool

# Здоровье
curl -s http://localhost:8000/health | python -m json.tool

# Версии индекса
curl -s http://localhost:8000/admin/versions | python -m json.tool
```

---

## Резюме: что проверять по порядку

| Шаг | Что делаешь | На что смотреть |
|-----|------------|-----------------|
| 1 | `POST /index` | Ответ: `document_count`, `chunk_count`, `index_version` |
| 2 | `GET /health` | Все подсистемы `true` |
| 3 | Поиск `mode: dense` | Синонимы в топе, семантическая близость |
| 4 | Поиск `mode: bm25` | Только точные совпадения слов |
| 5 | Поиск `mode: hybrid` | Слияние двух выдач |
| 6 | Поиск `mode: hybrid_rerank` | Порядок отличается от hybrid |
| 7 | Поиск с `must_contain` | Только документы с указанными словами |
| 8 | Поиск с `must_exclude` | Указанные документы исключены |
| 9 | Поиск с опечаткой | Поле `typo_suggestion` в ответе |
| 10 | Поиск с аббревиатурой | Поле `expanded_query` в ответе |
| 11 | `GET /admin/versions` | Список версий со статусами |
| 12 | `POST /reindex` | Новая версия, старая стала `superseded` |
| 13 | `POST /admin/rollback/{v}` | Активная версия переключена |
| 14 | `data/logs/queries.jsonl` | Каждый поиск записан в лог |
| 15 | `http://localhost:8000/docs` | Swagger — все эндпоинты с формами |
