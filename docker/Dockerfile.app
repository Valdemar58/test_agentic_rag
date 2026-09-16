# syntax=docker/dockerfile:1.7
# Образ приложения: MCP-сервер, мок сервиса карточек (и UI с этапа 7). Внешний код SDK и сервиса
# карточек в образ не попадает — монтируется томами (§8.0 ТЗ). torch — сборка CPU: в рантайме
# эмбеддинги и reranker считаются на CPU (§2), инжест с GPU выполняется на хосте.
FROM python:3.13-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    VIRTUAL_ENV=/app/.venv \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    APP_CONFIG_PATH=/app/configs/app.yaml

COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /uvx /bin/

WORKDIR /app

# torch из uv.lock (сборка CUDA для Linux) и его CUDA-библиотеки в CPU-образ не ставятся;
# torch CPU ставится отдельно из индекса PyTorch той же версии, что в lock.
ARG SKIP_PACKAGES="torch torchvision triton cuda-bindings cuda-pathfinder cuda-toolkit \
    nvidia-cublas nvidia-cuda-cupti nvidia-cuda-nvrtc nvidia-cuda-runtime nvidia-cudnn-cu13 \
    nvidia-cufft nvidia-cufile nvidia-curand nvidia-cusolver nvidia-cusparse nvidia-cusparselt-cu13 \
    nvidia-nccl-cu13 nvidia-nvjitlink nvidia-nvshmem-cu13 nvidia-nvtx"
ARG TORCH_CPU="torch==2.14.0 torchvision==0.29.0"

# Слой зависимостей: только файлы проекта, чтобы кэшироваться независимо от исходников.
COPY pyproject.toml uv.lock README.md ./
COPY tools/tessa_export/pyproject.toml tools/tessa_export/README.md tools/tessa_export/
RUN --mount=type=cache,target=/root/.cache/uv \
    skip=""; for name in $SKIP_PACKAGES; do skip="$skip --no-install-package $name"; done \
    && uv sync --frozen --no-dev --no-install-workspace $skip

COPY src src
COPY mocks mocks
COPY configs configs
COPY tools/tessa_export tools/tessa_export
# Второй sync ставит проект и workspace-пакет; torch CPU — после него: sync приводит окружение
# к lock и удалил бы пакет, поставленный отдельно.
RUN --mount=type=cache,target=/root/.cache/uv \
    skip=""; for name in $SKIP_PACKAGES; do skip="$skip --no-install-package $name"; done \
    && uv sync --frozen --no-dev $skip \
    && uv pip install --index-url https://download.pytorch.org/whl/cpu $TORCH_CPU

ENV PATH="/app/.venv/bin:$PATH"
CMD ["python", "-m", "mcp_server"]
