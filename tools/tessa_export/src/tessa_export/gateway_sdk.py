"""Шлюз к Тессе на SDK tessa_client.

Карточка запрашивается одним POST /api/v1/cards/get: сырой JSON сохраняется как есть, типизированная
модель Card разбирается публичными средствами SDK, а CardData строится тем же путём, что и в сервисе
карточек заказчика (Card.model_dump() → CardData.model_validate). Файлы скачиваются через
CardsResource.get_file_content.
"""

from __future__ import annotations

import contextlib
import io
import time
from typing import Any
from uuid import UUID

import httpx

from tessa_export.config import ExportConfig
from tessa_export.external import attach_external_code
from tessa_export.models import (
    INCOMING_SECTION,
    OUTGOING_SECTION,
    CardAccessError,
    CardNotFoundError,
    CardSnapshot,
    DownloadedContent,
    FileInfo,
    GatewayConnectionError,
    GatewayError,
    SectionSnapshot,
    parse_links,
)

CARDS_GET_PATH = "/api/v1/cards/get"
_RETRY_DELAY_SECONDS = 0.5


class SdkGateway:
    """Реализация TessaGateway поверх tessa_client и схем сервиса карточек."""

    def __init__(self, config: ExportConfig, username: str, password: str) -> None:
        attach_external_code(config.external)

        from robot_skills.core.cards.schemas import CardData
        from tessa_client import exceptions as sdk_exceptions
        from tessa_client.auth import TessaAuth
        from tessa_client.models.card_requests import CardGetRequest
        from tessa_client.models.card_responses import CardGetResponse
        from tessa_client.models.enums import CardGetMode, ValidationResultType
        from tessa_client.resources.cards import CardsResource
        from tessa_client.typed_json import denormalize_tessa_json, normalize_tessa_json

        self._card_data_cls: Any = CardData
        self._exc: Any = sdk_exceptions
        self._get_request_cls: Any = CardGetRequest
        self._get_response_cls: Any = CardGetResponse
        self._read_only_mode: Any = CardGetMode.READ_ONLY
        self._error_type: Any = ValidationResultType.ERROR
        self._normalize: Any = normalize_tessa_json
        self._denormalize: Any = denormalize_tessa_json
        self._max_retries = config.tessa.max_retries

        tessa = config.tessa
        verify: bool | str = str(tessa.ca_bundle) if tessa.ca_bundle else tessa.verify_tls
        self._auth: Any = TessaAuth(
            base_url=tessa.base_url,
            username=username,
            password=password,
            verify=verify,
            timeout=tessa.timeout_seconds,
            tessa_version=tessa.tessa_version,
        )
        self._session = httpx.Client(
            base_url=tessa.base_url,
            auth=self._auth,
            timeout=tessa.timeout_seconds,
            verify=verify,
            headers={"Content-Type": "application/json"},
        )
        self._cards: Any = CardsResource(self._session, max_retries=tessa.max_retries)

    def check_connection(self) -> None:
        """Открывает сессию заранее, чтобы ошибка логина или сети была видна сразу и понятно."""
        try:
            self._auth.login()
        except self._exc.TessaAuthenticationError as exc:
            raise CardAccessError(f"Тесса отклонила логин/пароль (HTTP {exc.status_code}): {exc}") from exc
        except (self._exc.TessaConnectionError, self._exc.TessaTimeoutError) as exc:
            raise GatewayConnectionError(f"сервер Тессы недоступен: {exc}") from exc
        except self._exc.TessaAPIError as exc:
            raise GatewayError(f"ошибка входа в Тессу (HTTP {exc.status_code}): {exc}") from exc

    def get_card(self, card_id: UUID) -> CardSnapshot:
        request = self._get_request_cls(card_id=card_id, get_mode=self._read_only_mode)
        body = self._denormalize(request.model_dump(mode="python", by_alias=True, exclude_none=True))
        response = self._post_with_retries(CARDS_GET_PATH, body, card_id)
        if response.status_code >= 400:
            raise self._map_api_error(self._exc.TessaAPIError.from_response(response), card_id)
        raw: dict[str, Any] = response.json()
        typed = self._get_response_cls.model_validate(self._normalize(raw))
        self._raise_if_operation_failed(typed, card_id)
        card = typed.card
        if card is None:
            raise CardNotFoundError(f"карточка {card_id}: сервер вернул пустую карточку")
        card_data = self._card_data_cls.model_validate(card.model_dump())
        return snapshot_from_card(card, raw, card_data.model_dump(mode="json"))

    def download_file(self, card_id: UUID, file: FileInfo) -> DownloadedContent:
        # SDK печатает тело каждого запроса в stdout; глушим, чтобы не засорять лог заказчика
        with contextlib.redirect_stdout(io.StringIO()):
            try:
                downloaded = self._cards.get_file_content(card_id, file.row_id, file.version_row_id)
            except self._exc.TessaAPIError as exc:
                raise self._map_api_error(exc, card_id, file) from exc
            except (self._exc.TessaConnectionError, self._exc.TessaTimeoutError) as exc:
                raise GatewayConnectionError(f"файл «{file.name}» карточки {card_id}: {exc}") from exc
        return DownloadedContent(
            content=downloaded.content,
            file_name=downloaded.file_name,
            content_type=downloaded.content_type,
        )

    def close(self) -> None:
        self._session.close()
        self._auth.close()

    def _post_with_retries(self, path: str, body: dict[str, Any], card_id: UUID) -> httpx.Response:
        attempt = 0
        while True:
            attempt += 1
            try:
                return self._session.post(path, json=body)
            except httpx.TimeoutException as exc:
                raise GatewayConnectionError(f"карточка {card_id}: тайм-аут запроса {path}") from exc
            except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
                if attempt > self._max_retries:
                    raise GatewayConnectionError(
                        f"карточка {card_id}: не удалось подключиться к Тессе ({exc})"
                    ) from exc
                time.sleep(_RETRY_DELAY_SECONDS * attempt)
            except httpx.RequestError as exc:
                raise GatewayConnectionError(f"карточка {card_id}: ошибка сети: {exc}") from exc

    def _raise_if_operation_failed(self, typed: Any, card_id: UUID) -> None:
        validation_result = getattr(typed, "validation_result", None)
        items = getattr(validation_result, "items", None) or []
        messages = [str(item.message) for item in items if item.type == self._error_type]
        if messages:
            raise GatewayError(f"карточка {card_id}: сервер сообщил об ошибке: {'; '.join(messages)}")

    def _map_api_error(self, exc: Any, card_id: UUID, file: FileInfo | None = None) -> GatewayError:
        subject = f"карточка {card_id}" if file is None else f"файл «{file.name}» карточки {card_id}"
        status = getattr(exc, "status_code", None)
        if isinstance(exc, self._exc.TessaNotFoundError):
            return CardNotFoundError(f"{subject}: не найдено (HTTP {status}): {exc}")
        if isinstance(exc, self._exc.TessaAuthenticationError | self._exc.TessaPermissionError):
            return CardAccessError(f"{subject}: нет доступа (HTTP {status}): {exc}")
        return GatewayError(f"{subject}: ошибка Тессы (HTTP {status}): {exc}")


def snapshot_from_card(card: Any, raw: dict[str, Any], card_data_json: dict[str, Any]) -> CardSnapshot:
    """Типизированная модель Card SDK → снимок карточки."""
    sections: dict[str, SectionSnapshot] = {}
    for name, section in (card.sections or {}).items():
        sections[name] = SectionSnapshot(
            fields=dict(section.fields) if section.fields is not None else None,
            rows=[dict(row) for row in section.rows] if section.rows is not None else None,
        )
    files = [
        FileInfo(
            row_id=item.row_id,
            version_row_id=item.version_row_id,
            name=item.name or "",
            size=int(item.size),
            version_number=int(item.version_number),
            category=item.category_caption,
            is_virtual=bool(item.is_virtual),
            type_name=item.type_name,
        )
        for item in card.files or []
    ]
    outgoing_rows = sections[OUTGOING_SECTION].rows if OUTGOING_SECTION in sections else None
    incoming_rows = sections[INCOMING_SECTION].rows if INCOMING_SECTION in sections else None
    return CardSnapshot(
        card_id=card.id,
        type_id=card.type_id,
        type_name=card.type_name,
        type_caption=card.type_caption,
        sections=sections,
        files=files,
        outgoing=parse_links(outgoing_rows),
        incoming=parse_links(incoming_rows),
        raw=raw,
        card_data_json=card_data_json,
    )
