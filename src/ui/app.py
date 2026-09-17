"""Chainlit-приложение (этап 7, FR-7): диалоги в PostgreSQL, стриминг, шаги агента, цитаты и 👍/👎.

Запуск: `uv run python -m ui` (хост и порт из `configs/app.yaml`, секция `ui`) или
`chainlit run src/ui/app.py`. Вход: в dev — один пользователь по `UI_USERNAME`/`UI_PASSWORD`; в проде —
OIDC, если заданы переменные `OAUTH_GENERIC_*` Chainlit (решение заказчика 2026-09-16). История
диалогов и фидбэк — в таблицах проекта через `ui.data_layer` (FR-9), трейс ответа связан с фидбэком
через `trace_id` сообщения (FR-8). Логики поверх фидбэка нет (§7 ТЗ).
"""

from __future__ import annotations

import asyncio
import logging
import os
import secrets
from functools import partial
from typing import Annotated, Any, Literal

import chainlit as cl
from chainlit.auth import get_current_user
from chainlit.config import config as chainlit_config
from chainlit.data.acl import is_thread_author
from chainlit.server import app as server
from chainlit.types import ThreadDict
from chainlit.utils import utc_now
from fastapi import Depends, HTTPException
from fastapi.responses import PlainTextResponse

from agent.llm import llm_ready
from agent.runner import AgentRunner, AgentSession
from agent.service import build_runner
from common.config import load_app_config
from common.logs import configure_logging
from common.settings import load_settings
from ui.data_layer import ELEMENT_CONTENT_ROUTE, AppDataLayer, parse_id
from ui.flow import (
    Final,
    Restart,
    StepFinished,
    StepStarted,
    Token,
    UiEvent,
    run_question,
    wait_until_ready,
)
from ui.persistence import ConversationStore
from ui.state import restore_session

logger = logging.getLogger("ui")

CONFIG = load_app_config()
SETTINGS = load_settings()
configure_logging(CONFIG.logging.level)
chainlit_config.ui.name = CONFIG.ui.title

SESSION_KEY = "agent_session"
STEP_TYPE: Literal["tool"] = "tool"
FEEDBACK_SCORE = "user_feedback"
CITATION_DISPLAY: Literal["side"] = "side"
CurrentUser = Annotated[Any, Depends(get_current_user)]
OIDC_ENV = (
    "OAUTH_GENERIC_CLIENT_ID",
    "OAUTH_GENERIC_CLIENT_SECRET",
    "OAUTH_GENERIC_AUTH_URL",
    "OAUTH_GENERIC_TOKEN_URL",
    "OAUTH_GENERIC_USER_INFO_URL",
    "OAUTH_GENERIC_SCOPES",
)
STAND_ERROR = "Стенд недоступен: {error}. Нужны MCP-сервер и vLLM профиля runtime."
ANSWER_ERROR = "Не удалось получить ответ: {error}"
LLM_KEY = "llm"
LLM_LOADING_TITLE = "Модель ещё загружается, жду готовности"
LLM_READY_TITLE = "Модель готова"
LLM_NOT_READY_TITLE = "Модель не готова"
LLM_NOT_READY = (
    "Модель ещё загружается или vLLM недоступен: ответить сейчас нельзя. Повторите вопрос через минуту."
)

_store = ConversationStore.from_settings(SETTINGS)
_runner: AgentRunner | None = None
_runner_lock = asyncio.Lock()


async def get_runner() -> AgentRunner:
    """Раннер агента общий для всех сессий (одно соединение с MCP, LLM по ролям); собирается лениво."""
    global _runner
    async with _runner_lock:
        if _runner is None:
            _runner = await build_runner(CONFIG, SETTINGS)
        return _runner


def oidc_configured(environ: dict[str, str] | os._Environ[str] = os.environ) -> bool:
    return all(environ.get(name) for name in OIDC_ENV)


# ---------- хранение и вход ----------


@cl.data_layer
def data_layer() -> AppDataLayer:
    return AppDataLayer(_store)


if SETTINGS.ui_password.get_secret_value():

    @cl.password_auth_callback
    async def login(username: str, password: str) -> Any:
        expected_user = SETTINGS.ui_username
        expected_password = SETTINGS.ui_password.get_secret_value()
        if secrets.compare_digest(username, expected_user) and secrets.compare_digest(
            password, expected_password
        ):
            return cl.User(identifier=username, metadata={"provider": "credentials"})
        return None


if oidc_configured():

    @cl.oauth_callback
    async def oauth(
        provider_id: str,
        token: str,
        raw_user_data: dict[str, str],
        default_user: Any,
        id_token: str | None = None,
    ) -> Any:
        return default_user


if not SETTINGS.ui_password.get_secret_value() and not oidc_configured():
    logger.warning(
        "Вход в UI выключен (UI_PASSWORD пуст, OAUTH_GENERIC_* не заданы): список диалогов недоступен"
    )


# ---------- жизненный цикл ----------


@cl.on_app_startup
async def startup() -> None:
    try:
        await get_runner()
    except Exception as exc:  # noqa: BLE001 — стенд может подниматься дольше UI; повтор при первом вопросе
        logger.warning("Агент не собран при старте UI: %s", exc)


@cl.on_app_shutdown
async def shutdown() -> None:
    if _runner is not None:
        await _runner.aclose()
    await _store.close()


@cl.on_chat_start
async def chat_start() -> None:
    cl.user_session.set(SESSION_KEY, AgentSession(CONFIG, session_id=cl.context.session.thread_id))


@cl.on_chat_resume
async def chat_resume(thread: ThreadDict) -> None:
    session = restore_session(CONFIG, str(thread["id"]), thread.get("steps") or [])
    cl.user_session.set(SESSION_KEY, session)
    logger.info("Диалог %s возобновлён: ходов %d", thread["id"], len(session.memory.turns))


# ---------- вопрос → шаги, стрим, ответ ----------


class ChainlitPresenter:
    """Переводит события потока в шаги и сообщение Chainlit."""

    def __init__(self) -> None:
        self._steps: dict[str, Any] = {}
        self._message: Any = None

    @staticmethod
    def _new_step(title: str) -> Any:
        # Родитель — run-шаг обработчика сообщения: шаги и ответ идут в порядке создания внутри него
        current = cl.context.current_step
        step = cl.Step(name=title, type=STEP_TYPE, parent_id=current.id if current is not None else None)
        step.start = utc_now()
        return step

    async def handle(self, event: UiEvent) -> None:
        if isinstance(event, StepStarted):
            step = self._new_step(event.title)
            await step.send()
            self._steps[event.key] = step
        elif isinstance(event, StepFinished):
            step = self._steps.pop(event.key, None)
            if step is None:
                step = self._new_step(event.title)
                await step.send()
            step.name = event.title
            if event.output:
                step.output = event.output
            step.is_error = not event.ok
            step.end = utc_now()
            await step.update()
        elif isinstance(event, Token):
            if self._message is None:
                self._message = cl.Message(content="")
            await self._message.stream_token(event.text)
        elif isinstance(event, Restart):
            if self._message is not None:
                self._message.content = ""
                await self._message.update()
        elif isinstance(event, Final):
            message = self._message if self._message is not None else cl.Message(content="")
            message.content = event.text
            message.elements = [
                cl.Text(name=citation.name, content=citation.content, display=CITATION_DISPLAY)
                for citation in event.citations
            ]
            message.metadata = event.metadata
            await message.send()
            self._message = message

    async def fail(self, text: str) -> None:
        for step in self._steps.values():
            step.is_error = True
            step.end = utc_now()
            await step.update()
        self._steps.clear()
        await cl.ErrorMessage(content=text).send()


@cl.on_message
async def on_message(message: Any) -> None:
    session = cl.user_session.get(SESSION_KEY)
    if session is None:
        session = AgentSession(CONFIG, session_id=cl.context.session.thread_id)
        cl.user_session.set(SESSION_KEY, session)
    try:
        runner = await get_runner()
    except Exception as exc:  # noqa: BLE001 — недоступный MCP или vLLM: сообщение вместо трассировки
        await cl.ErrorMessage(content=STAND_ERROR.format(error=exc)).send()
        return
    presenter = ChainlitPresenter()
    try:
        if not await _ensure_llm_ready(presenter):
            return
        async for event in run_question(runner, session, str(message.content)):
            await presenter.handle(event)
    except Exception as exc:  # noqa: BLE001 — ошибка одного вопроса не должна ронять сессию
        logger.exception("Ошибка при ответе на вопрос")
        await presenter.fail(ANSWER_ERROR.format(error=exc))


async def _ensure_llm_ready(presenter: ChainlitPresenter) -> bool:
    """После перезапуска стенда vLLM грузит модель дольше, чем поднимается UI: ждём шагом, а не ошибкой."""
    probe = partial(llm_ready, CONFIG, SETTINGS, timeout_s=CONFIG.ui.llm_ready_poll_s)
    if await probe():
        return True
    await presenter.handle(StepStarted(key=LLM_KEY, title=LLM_LOADING_TITLE))
    ready = await wait_until_ready(
        probe, timeout_s=CONFIG.ui.llm_ready_wait_s, poll_s=CONFIG.ui.llm_ready_poll_s
    )
    await presenter.handle(
        StepFinished(key=LLM_KEY, title=LLM_READY_TITLE if ready else LLM_NOT_READY_TITLE, ok=ready)
    )
    if not ready:
        await cl.ErrorMessage(content=LLM_NOT_READY).send()
    return ready


# ---------- фидбэк (FR-7): запись в БД делает data layer, здесь — score в Langfuse ----------


@cl.on_feedback
async def on_feedback(feedback: Any) -> None:
    trace_id = await _store.message_trace_id(parse_id(feedback.forId))
    if not trace_id:
        logger.info("Фидбэк %s к сообщению без трейса: score в Langfuse не пишется", feedback.value)
        return
    if _runner is None:
        logger.warning("Фидбэк к трейсу %s получен до сборки агента: score не записан", trace_id)
        return
    _runner.tracing.score(
        trace_id, name=FEEDBACK_SCORE, value=float(feedback.value), comment=feedback.comment
    )


# ---------- текст процитированного фрагмента из БД (для возобновлённых диалогов) ----------


async def element_content(thread_id: str, element_id: str, current_user: CurrentUser) -> PlainTextResponse:
    if current_user is not None:
        await is_thread_author(current_user.identifier, thread_id)
    try:
        element = await _store.get_element(parse_id(thread_id), parse_id(element_id))
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Элемент не найден") from exc
    if element is None or element.content is None:
        raise HTTPException(status_code=404, detail="Элемент не найден")
    return PlainTextResponse(element.content, media_type="text/plain; charset=utf-8")


# Маршрут ставится перед остальными: у Chainlit последним стоит catch-all «/{full_path}», который отдаёт
# index.html на любой путь, а маршруты FastAPI сопоставляются по порядку регистрации
server.add_api_route(ELEMENT_CONTENT_ROUTE, element_content, methods=["GET"])
server.router.routes.insert(0, server.router.routes.pop())
