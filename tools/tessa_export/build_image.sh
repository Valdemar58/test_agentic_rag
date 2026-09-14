#!/usr/bin/env bash
# Сборка образа tessa-export и выгрузка в архив для передачи заказчику (Linux/macOS).
# Использование: tools/tessa_export/build_image.sh [версия]
# Пути к внешнему коду: TESSA_SDK_PATH и CARD_SERVICE_PATH (переменные окружения или .env).
set -euo pipefail

VERSION="${1:-0.1.1}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

if [[ -f .env ]]; then
  set -a; # shellcheck disable=SC1091
  source .env; set +a
fi
: "${TESSA_SDK_PATH:?Задайте TESSA_SDK_PATH}"
: "${CARD_SERVICE_PATH:?Задайте CARD_SERVICE_PATH}"

IMAGE="tessa-export:${VERSION}"
docker build -f tools/tessa_export/Dockerfile \
  --build-context "tessa_sdk=${TESSA_SDK_PATH}" \
  --build-context "card_service=${CARD_SERVICE_PATH}" \
  -t "${IMAGE}" .

DIST="${ROOT}/dist/tessa-export-${VERSION}"
mkdir -p "${DIST}"
docker save "${IMAGE}" -o "${DIST}/tessa-export-${VERSION}.tar"
cp tools/tessa_export/config.example.yaml "${DIST}/config.yaml"
cp tools/tessa_export/seed_cards.yaml "${DIST}/"
cp tools/tessa_export/README.md "${DIST}/README.md"
cp tools/tessa_export/tessa.env.example "${DIST}/tessa.env"
echo "Готово: ${DIST}"
ls -la "${DIST}"
