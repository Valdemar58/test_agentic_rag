"""Трейсинг в Langfuse (6.6, FR-8): каждый вопрос — трейс с цепочкой переписывание → поиски → ответ.

LLM-вызовы и вызовы инструментов LlamaIndex попадают в трейс через OpenInference-инструментацию
(`openinference-instrumentation-llama-index`): она пишет в глобальный OpenTelemetry-провайдер, который
создаёт клиент Langfuse. Спаны раннера — этапы вопроса `rewrite`, `tool_loop`, `answer` — и атрибуты
трейса (идентификатор сессии диалога, вопрос, ответ) добавляются здесь. Без `LANGFUSE_ENABLED`
работает заглушка `NoopTracing`, а `trace_id` в ответе пуст.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from typing import Any, Literal, Protocol, cast

from common.config import LangfuseSettings
from common.settings import Settings

logger = logging.getLogger(__name__)

StepKind = Literal["chain", "agent", "retriever", "generation", "span", "tool"]
QUESTION_OBSERVATION = "question"


class StepHandle(Protocol):
    def update(self, *, output: Any = None, metadata: dict[str, Any] | None = None) -> None: ...


class QuestionHandle(Protocol):
    @property
    def trace_id(self) -> str | None: ...

    def update(self, *, output: Any = None, metadata: dict[str, Any] | None = None) -> None: ...


class Tracing(Protocol):
    """Контракт трейсинга раннера; реализации — Langfuse и заглушка."""

    @property
    def enabled(self) -> bool: ...

    def question(self, question: str, *, session_id: str) -> AbstractContextManager[QuestionHandle]: ...

    def step(self, name: str, *, kind: StepKind, input: Any = None) -> AbstractContextManager[StepHandle]: ...

    def score(self, trace_id: str, *, name: str, value: float, comment: str | None = None) -> None:
        """Оценка трейса (👍/👎 пользователя, FR-7): score Langfuse, привязанный к трейсу ответа."""
        ...

    def flush(self) -> None: ...


class _NoopHandle:
    trace_id: str | None = None

    def update(self, *, output: Any = None, metadata: dict[str, Any] | None = None) -> None:
        return None


class NoopTracing:
    """Трейсинг выключен: обработчики ничего не делают."""

    enabled = False

    @contextmanager
    def question(self, question: str, *, session_id: str) -> Iterator[QuestionHandle]:
        yield _NoopHandle()

    @contextmanager
    def step(self, name: str, *, kind: StepKind, input: Any = None) -> Iterator[StepHandle]:
        yield _NoopHandle()

    def score(self, trace_id: str, *, name: str, value: float, comment: str | None = None) -> None:
        return None

    def flush(self) -> None:
        return None


class _LangfuseSpanHandle:
    def __init__(self, span: Any, trace_id: str | None = None) -> None:
        self._span = span
        self.trace_id = trace_id

    def update(self, *, output: Any = None, metadata: dict[str, Any] | None = None) -> None:
        self._span.update(output=output, metadata=metadata)


_instrumented = False


def instrument_llama_index() -> None:
    """Подключает OpenInference-инструментацию LlamaIndex к глобальному провайдеру OTel (один раз)."""
    global _instrumented
    if _instrumented:
        return
    from openinference.instrumentation.llama_index import LlamaIndexInstrumentor

    LlamaIndexInstrumentor().instrument()
    _instrumented = True


class LangfuseTracing:
    """Трейсы в self-hosted Langfuse (FR-8, AC-8.1)."""

    enabled = True

    def __init__(self, client: Any) -> None:
        self._client = client
        instrument_llama_index()

    @classmethod
    def from_settings(cls, settings: Settings, config: LangfuseSettings) -> LangfuseTracing:
        from langfuse import Langfuse

        client = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key.get_secret_value(),
            base_url=settings.langfuse_url,
            environment=config.environment,
            flush_interval=config.flush_interval_s,
        )
        logger.info("Трейсинг Langfuse включён: %s", settings.langfuse_url)
        return cls(client)

    @contextmanager
    def question(self, question: str, *, session_id: str) -> Iterator[QuestionHandle]:
        from langfuse import propagate_attributes

        client = cast(Any, self._client)
        with client.start_as_current_observation(
            name=QUESTION_OBSERVATION, as_type="agent", input={"question": question}
        ) as span:
            with propagate_attributes(session_id=session_id, trace_name=QUESTION_OBSERVATION):
                yield _LangfuseSpanHandle(span, trace_id=client.get_current_trace_id())

    @contextmanager
    def step(self, name: str, *, kind: StepKind, input: Any = None) -> Iterator[StepHandle]:
        client = cast(Any, self._client)
        with client.start_as_current_observation(name=name, as_type=kind, input=input) as span:
            yield _LangfuseSpanHandle(span)

    def score(self, trace_id: str, *, name: str, value: float, comment: str | None = None) -> None:
        client = cast(Any, self._client)
        client.create_score(trace_id=trace_id, name=name, value=value, data_type="NUMERIC", comment=comment)
        client.flush()

    def flush(self) -> None:
        self._client.flush()


def build_tracing(settings: Settings, config: LangfuseSettings) -> Tracing:
    if not settings.langfuse_enabled:
        return NoopTracing()
    if not settings.langfuse_public_key or not settings.langfuse_secret_key.get_secret_value():
        logger.warning(
            "LANGFUSE_ENABLED задан без ключей LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY: трейсинг выключен"
        )
        return NoopTracing()
    return LangfuseTracing.from_settings(settings, config)
