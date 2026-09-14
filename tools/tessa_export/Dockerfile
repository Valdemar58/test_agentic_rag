# syntax=docker/dockerfile:1.7
# Образ экспорт-скрипта tessa-export. Собирается исполнителем из корня репозитория:
#   docker build -f tools/tessa_export/Dockerfile \
#     --build-context tessa_sdk=<TESSA_SDK_PATH> --build-context card_service=<CARD_SERVICE_PATH> \
#     -t tessa-export:<версия> .
# SDK Тессы и схемы сервиса карточек попадают в образ через именованные build-context'ы
# и в репозиторий не копируются (§8.0 ТЗ).

FROM python:3.13-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:$PATH

COPY --from=ghcr.io/astral-sh/uv:0.12.1 /uv /uvx /bin/

WORKDIR /app

# Workspace uv: корневой pyproject + lock нужны для разрешения, ставится только пакет tessa-export
COPY pyproject.toml uv.lock README.md ./
COPY tools/tessa_export/pyproject.toml tools/tessa_export/README.md tools/tessa_export/
COPY tools/tessa_export/src tools/tessa_export/src
# --package ограничивает установку пакетом tessa-export и его зависимостями; корневой проект не ставится
RUN mkdir -p src && uv sync --locked --no-dev --no-editable --package tessa-export

# Внешний код по фиксированным путям; они же прописаны в config.example.yaml
COPY --from=tessa_sdk src /opt/external/tessa_sdk/src
COPY --from=card_service src /opt/external/robot_skills/src

# Конфиг-шаблон и seed-список внутри образа для справки; рабочий конфиг монтируется в /config
COPY tools/tessa_export/config.example.yaml tools/tessa_export/seed_cards.yaml /app/defaults/

RUN mkdir -p /config /output
VOLUME ["/config", "/output"]

ENTRYPOINT ["tessa-export"]
CMD ["--help"]
