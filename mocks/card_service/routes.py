"""Маршруты мока — подмножество реального сервиса карточек с его же схемами и форматом ошибок.

Модуль импортирует `robot_skills` напрямую, поэтому подключается только после
`ensure_external_paths` (см. `mocks.card_service.app.create_app`). Реализованы маршруты, нужные
RAG и контрактному тесту AC-2.4: `GET /health`, `POST /core/cards/get`,
`POST /core/cards/get-file-content`, `GET /core/card-types`, `GET /core/card-types/{type_id}`,
`POST /core/views/{view_alias}/get-data` (только представление типов связей). Basic auth
обязателен, как у реального сервиса, но принимаются любые учётные данные (N6).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Annotated, Any, cast
from urllib.parse import quote
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from robot_skills.core.card_types.schemas import CardTypeOut
from robot_skills.core.cards.schemas import CardData, CardGetFileContentIn, CardGetIn
from robot_skills.core.views.schemas import ViewGetDataIn, ViewResultOut
from robot_skills.errors import ErrorResponse
from robot_skills.exceptions import CoreResourceNotFoundError

from common.config import MockCardServiceSettings
from mocks.card_service.store import CardStore

REQUEST_ID_HEADER = "x-request-id"
OCTET_STREAM = "application/octet-stream"

basic_auth = HTTPBasic(realm="tessa")


def _store(request: Request) -> CardStore:
    return cast(CardStore, request.app.state.store)


def _settings(request: Request) -> MockCardServiceSettings:
    return cast(MockCardServiceSettings, request.app.state.settings)


def _request_id(request: Request) -> str:
    return str(getattr(request.state, "request_id", ""))


StoreDep = Annotated[CardStore, Depends(_store)]
SettingsDep = Annotated[MockCardServiceSettings, Depends(_settings)]
# Любая учётка принимается; без заголовка Authorization — 401, как у реального сервиса
AuthDep = Annotated[HTTPBasicCredentials, Depends(basic_auth)]

router = APIRouter()


@router.post("/cards/get", response_model=CardData)
async def get_card(store: StoreDep, _: AuthDep, body: CardGetIn) -> CardData:
    card = store.card(body.card_id) if body.card_id is not None else None
    if card is None:
        raise CoreResourceNotFoundError("card", str(body.card_id or body.card_type_id or body.card_type_name))
    return card


@router.post("/cards/get-file-content")
async def get_file_content(store: StoreDep, _: AuthDep, body: CardGetFileContentIn) -> Response:
    stored = store.file(body.card_id, body.file_id)
    if stored is None:
        raise CoreResourceNotFoundError("file", f"{body.card_id}/{body.file_id}")
    if stored.path is None:
        reason = stored.skipped_reason or "не скачан"
        raise CoreResourceNotFoundError(
            "file content", f"{body.file_id} (в архиве нет содержимого: {reason})"
        )
    if stored.version_row_id is not None and body.version_row_id != stored.version_row_id:
        raise CoreResourceNotFoundError(
            "file version",
            f"{body.version_row_id} (в архиве только последняя версия {stored.version_row_id})",
        )
    headers = {"Content-Disposition": f"attachment; filename*=UTF-8''{quote(stored.name, safe='')}"}
    return Response(
        content=stored.path.read_bytes(), media_type=stored.content_type or OCTET_STREAM, headers=headers
    )


@router.get("/card-types", response_model=list[CardTypeOut])
async def list_card_types(store: StoreDep, _: AuthDep) -> list[CardTypeOut]:
    return [CardTypeOut(id=item.id, name=item.name, caption=item.caption) for item in store.card_types]


@router.get("/card-types/{type_id}", response_model=CardTypeOut)
async def get_card_type(store: StoreDep, _: AuthDep, type_id: UUID) -> CardTypeOut:
    item = store.card_type(type_id)
    if item is None:
        raise CoreResourceNotFoundError("card_type", str(type_id))
    return CardTypeOut(id=item.id, name=item.name, caption=item.caption)


@router.post("/views/{view_alias}/get-data", response_model=ViewResultOut)
async def get_view_data(
    store: StoreDep, settings: SettingsDep, _: AuthDep, view_alias: str, body: ViewGetDataIn
) -> ViewResultOut:
    """Представление типов связей из связей экспорта; других представлений у мока нет (N20)."""
    if view_alias != settings.ref_type_view:
        raise CoreResourceNotFoundError("view", view_alias)
    rows: list[list[Any]] = [
        [str(item.id) if item.id else None, item.name, item.reverse_name] for item in store.ref_types
    ]
    page = rows
    if body.page_limit:
        offset = max((body.page_offset or 1) - 1, 0) * body.page_limit
        page = rows[offset : offset + body.page_limit]
    return ViewResultOut(
        row_count=len(rows) if body.calculate_row_counting else 0,
        columns=list(settings.ref_type_columns),
        scheme_types=None,
        rows=page,
    )


async def _not_found(request: Request, exc: Exception) -> JSONResponse:
    body = ErrorResponse(error="resource_not_found", message=str(exc), request_id=_request_id(request))
    return JSONResponse(status_code=404, content=body.model_dump())


async def _validation_error(request: Request, exc: Exception) -> JSONResponse:
    errors = cast(RequestValidationError, exc).errors()
    body = ErrorResponse(
        error="request_validation_error",
        message="Ошибка валидации запроса",
        request_id=_request_id(request),
        validation_items=[dict(item) for item in errors],
    )
    return JSONResponse(status_code=422, content=body.model_dump())


def build_app(store: CardStore, settings: MockCardServiceSettings) -> FastAPI:
    app = FastAPI(title="robot-skills (mock)")
    app.state.store = store
    app.state.settings = settings

    @app.middleware("http")
    async def request_context(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request.state.request_id = request.headers.get(REQUEST_ID_HEADER) or str(uuid4())
        response = await call_next(request)
        response.headers[REQUEST_ID_HEADER] = request.state.request_id
        return response

    app.add_exception_handler(CoreResourceNotFoundError, _not_found)
    app.add_exception_handler(RequestValidationError, _validation_error)

    @app.get("/health", tags=["health"])
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    app.include_router(router, prefix="/core", tags=["core"])
    return app
