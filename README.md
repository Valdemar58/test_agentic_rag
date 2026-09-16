# Агентский RAG по документам СЭД Тесса — MVP

Реализация по `TZ_agentic_rag_mvp.md`. Ход работ — в `PLAN.md`, правила — в `CLAUDE.md`.

> Статус: этап 5 (мок сервиса карточек и MCP-сервер). Реальный корпус получен экспорт-скриптом
> (этап 2), проиндексирован (этап 4) и лежит вне репозитория; для тестов используется синтетический
> корпус (`scripts/make_synthetic_corpus.py`). **Данные синтетические**; метрики M1–M8 снимаются
> только на реальном корпусе.

## Требования

- Python 3.13, [uv](https://docs.astral.sh/uv/)
- Docker + docker compose с NVIDIA Container Toolkit (драйвер ≥ 575 для образа vLLM CUDA 12.9)
- GPU: целевая — RTX 3080 12 ГиБ; бюджет VRAM 12 ГиБ соблюдается на любой карте
  (`gpu` в `configs/app.yaml`, доля памяти считается скриптом стенда)
- Память VM Docker Desktop: 16 ГиБ (лимиты контейнеров в `docker-compose.yml`)

## Быстрый старт

```
uv sync
cp .env.example .env               # TESSA_SDK_PATH, CARD_SERVICE_PATH, при необходимости порты
uv run python scripts/check.py     # ruff + format + mypy + pytest (или make check)
```

## Стенд (этап 3)

```
uv run python scripts/download_models.py        # веса по pinned-ревизиям в models/ (разово, ~16 ГиБ)
uv run python scripts/stack.py up runtime        # qdrant, postgres-app, vllm-qwen (Qwen3-8B AWQ)
uv run alembic upgrade head                      # схема прикладной БД
uv run python scripts/smoke.py                   # БД и миграции, Qdrant, LLM по-русски, tool calling
uv run python scripts/stack.py up ingest --switch   # dots.mocr вместо Qwen (профили взаимоисключены)
uv run python scripts/stack.py up runtime --observability   # + Langfuse (http://localhost:3000)
uv run python scripts/stack.py down
```

- `stack.py` вычисляет `--gpu-memory-utilization` из бюджета в гибибайтах и фактического объёма карты,
  поэтому vLLM получает одинаковый объём на RTX 3080 и на карте разработки.
- Профили `runtime` (Qwen3-8B) и `ingest` (dots.mocr) нельзя поднять одновременно: `up` второго
  завершается ошибкой, `--switch` останавливает первый.
- Все параметры — в `configs/app.yaml`; секреты, URL и порты — в `.env` (см. `.env.example`).
  Порты инфраструктуры публикуются только на 127.0.0.1; если 5432 занят, задайте `APP_DB_PORT`.
- Рантайм офлайн: `HF_HUB_OFFLINE=1`, телеметрия vLLM, Qdrant и Langfuse выключена.

## Инжест (этап 4)

```
uv run python scripts/run_ingest.py inspect --corpus data/<архив>   # состав, роли файлов, маршруты разбора
uv run python scripts/run_ingest.py run --corpus data/<архив>       # разбор (dots.mocr на GPU) → эмбеддинги → Qdrant
```

Инжест сам переключает GPU-профили: поднимает `vllm-dots` на время разбора сканов, затем считает
эмбеддинги bge-m3 на освободившейся карте. Повторный запуск обрабатывает только новые и изменённые
файлы (реестр в PostgreSQL). Отчёт — `data/work/reports/`.

## Мок сервиса карточек и MCP-сервер (этап 5)

```
uv run python -m mocks.card_service --corpus data/<архив>   # мок сервиса карточек, порт 8010
uv run python -m mcp_server                                 # MCP-сервер (streamable-http), http://127.0.0.1:8020/mcp
```

- Мок реализует маршруты и pydantic-схемы реального сервиса карточек заказчика (`robot_skills`
  по `CARD_SERVICE_PATH`) и отдаёт карточки из архива экспорта. Переключение мок ↔ реальный сервис —
  только `CARD_SERVICE_URL` (и учётка `CARD_SERVICE_USERNAME`/`CARD_SERVICE_PASSWORD`).
- MCP-сервер даёт агенту пять инструментов: `hybrid_search` (dense + sparse → RRF → reranker на CPU),
  `get_document_card`, `get_related_documents`, `get_document_content`, `glossary_lookup`
  (заглушка до этапа 8). Веса bge-m3 и reranker'а читаются из `models/`, работа офлайн.
- В Docker те же сервисы — `mcp-server` (профиль `runtime`) и `mock-card-service` (профиль `mock`,
  `stack.py up runtime` включает его по умолчанию, `--no-mock` — для стенда с реальным сервисом).
  Образ — `docker/Dockerfile.app`; внешний код монтируется томами из `TESSA_SDK_PATH` и
  `CARD_SERVICE_PATH`, архив — из `CORPUS_DIR`, адрес сервиса карточек для контейнера —
  `CARD_SERVICE_URL_DOCKER` (по умолчанию мок).

## Синтетический корпус

```
uv run python scripts/make_synthetic_corpus.py   # data/corpus/: 13 карточек, 28 файлов, отчёт валидации
```

Формат — тот же, что у реального экспорта (`export/cards`, `cards_raw`, `files`, `manifest.json`,
`links_graph.json`, `validation_report.md`); карточки валидны по схеме `CardData` сервиса карточек.
Каталог с реальным экспортом скрипт не перезаписывает.

## Внешний код

Код SDK Тессы и FastAPI-сервиса карточек заказчика в репозиторий не копируется (§8.0 ТЗ).
Пути задаются в `.env` (`TESSA_SDK_PATH`, `CARD_SERVICE_PATH`); pydantic-схемы карточек
импортируются как внешний пакет через `src/contracts/`. Без путей зависящие от них тесты скипаются.

Экспорт-скрипт для контура заказчика: `tools/tessa_export/README.md`.

Полная инструкция от установки до снятия метрик появится на этапе 10.
